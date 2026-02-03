import torch
import torch.nn.functional as F

from transformers import Trainer
from transformers.trainer import (
    ###
    _is_peft_model,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    is_torch_xla_available,
    SaveStrategy
)
from typing import List, Optional, Dict
from utils.gem_triton_loss import GEMLoss
from entmax import entmax15, sparsemax, sparsemax_loss, entmax15_loss
import torch.distributed as dist
import os


class SFTTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.grad_log=[]
        self.recorded = {}

    @torch.no_grad
    def compute_training_logs(self, logits, labels):
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        training_logs = {}
        if self.args.print_entropy:
            entropy = chunked_entropy_from_logits(
                shift_logits,
                batch_size=max(1, shift_logits.size(0) // 4),
            ).mean()
            training_logs["entropy"] = round(entropy.item(), 2)

        return training_logs

    def ts2_loss(self, logits, labels, alpha, temperature=1.0, num_items_in_batch=None):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        sparse_probs = sparsemax(shift_logits * temperature, dim=-1)

        s_loss = sparsemax_loss(shift_logits * temperature, shift_labels)

        softmax_probs = F.softmax(shift_logits, dim=-1)

        one_hot = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()

        neg_mask = (sparse_probs == 0.0) & (~one_hot)

        suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
        suppressed_mass = torch.clamp(suppressed_mass, max=0.99)

        ns_loss = -torch.log(1.0 - suppressed_mass)

        # Combine total loss
        loss = s_loss + alpha * ns_loss

        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.mean()
        return loss

    def ts2_adaptive(self, logits, labels, alpha, temperature=1.0, num_items_in_batch=None):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        sparse_probs = sparsemax(shift_logits * temperature, dim=-1)

        s_loss = sparsemax_loss(shift_logits * temperature, shift_labels)

        softmax_probs = F.softmax(shift_logits, dim=-1)

        one_hot = F.one_hot(shift_labels, num_classes=softmax_probs.size(-1)).bool()

        neg_mask = (sparse_probs == 0.0) & (~one_hot)

        suppressed_mass = (softmax_probs * neg_mask.float()).sum(dim=-1)
        suppressed_mass = torch.clamp(suppressed_mass, max=0.99)

        ns_loss = -torch.log(1.0 - suppressed_mass)

        target_sparse = sparse_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1).detach()

        # Combine total loss
        loss = s_loss + alpha * (1.0 - target_sparse) * ns_loss

        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss.mean()

        return loss

    def gem_loss(self, logits, labels, num_items_in_batch, beta=0.7, ignore_index=-100, h="logsigmoid"):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        # shift_logits: [N, V]
        max_logits = shift_logits.max(dim=-1).values
        if (max_logits == float('-inf')).any():
            raise ValueError("All logits for some samples are -inf or extremely small, which breaks sparsemax.")

        # 调试输出 support size 计算前的 logits
        print("sparsemax logits max:", max_logits.max().item(), "min:", max_logits.min().item())

        with torch.no_grad():
            logits_on_labels = torch.gather(
                shift_logits, dim=-1, index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            logits_diff = shift_logits - logits_on_labels.unsqueeze(-1)
            if h == "linear":
                weights = torch.ones_like(logits_diff)
            elif h == "logsigmoid":
                weights = F.sigmoid(0.01 * logits_diff)
            else:
                raise ValueError(h)

        gene_log_probs = F.log_softmax(shift_logits, dim=-1)
        q_probs = torch.exp(F.log_softmax(shift_logits / beta, dim=-1)).detach()

        real_log_probs = torch.gather(
            gene_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        )

        if num_items_in_batch is not None:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).sum() / num_items_in_batch
        else:
            loss = -torch.sum(
                q_probs * weights * (real_log_probs - gene_log_probs), dim=-1
            ).mean()

        return loss

    def gem_loss_triton(self, logits, labels, num_items_in_batch, beta=0.7, ignore_index=-100, h="linear"):
        assert h == "linear", "Only linear is supported for gem_loss_triton for now."

        if num_items_in_batch is not None:
            gem_loss_func = GEMLoss(beta=beta, ignore_index=ignore_index, reduction="none")
        else:
            gem_loss_func = GEMLoss(beta=beta, ignore_index=ignore_index, reduction="mean")

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        mask = shift_labels != -100
        shift_logits = shift_logits[mask]
        shift_labels = shift_labels[mask]

        loss = gem_loss_func(shift_logits, shift_labels)

        if num_items_in_batch is not None:
            loss = loss.sum() / num_items_in_batch
        else:
            loss = loss
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )

            if self.args.loss == "ce" or self.args.loss == "neft" or self.control.should_evaluate:
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            elif self.args.loss == "gem":
                loss = self.gem_loss(
                    outputs.logits,
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch,
                    beta=self.args.gem_beta,
                    h=self.args.gem_h
                )
            elif self.args.loss == "gem_triton":
                loss = self.gem_loss_triton(
                    outputs.logits,
                    inputs["labels"],
                    num_items_in_batch=num_items_in_batch,
                    beta=self.args.gem_beta,
                    h=self.args.gem_h
                )
            elif self.args.loss == "ts2":
                loss = self.ts2_loss(
                    outputs.logits,
                    inputs["labels"],
                    alpha=self.args.ts2_alpha,
                    temperature=self.args.ts2_temperature,
                    num_items_in_batch=num_items_in_batch
                )
            elif self.args.loss == "ts2_adaptive":
                loss = self.ts2_adaptive(
                    outputs.logits,
                    inputs["labels"],
                    alpha=self.args.ts2_alpha,
                    temperature=self.args.ts2_temperature,
                    num_items_in_batch=num_items_in_batch
                )

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        # logs
        if not self.control.should_evaluate:
            self.training_logs = self.compute_training_logs(
                outputs.logits, inputs["labels"]
            )
            self.training_logs["ce_loss"] = (
                outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            )
            self.training_logs["ce_loss"] = round(self.training_logs["ce_loss"].item(), 4)

        return (loss, outputs) if return_outputs else loss

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time,
                                 learning_rate=None):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = self._get_learning_rate()
            if getattr(self, "training_logs", None):
                logs.update(self.training_logs)

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs, start_time)

        metrics = None
        if self.control.should_evaluate:
            metrics = self._evaluate(trial, ignore_keys_for_eval)
            is_new_best_metric = self._determine_best_metric(metrics=metrics, trial=trial)

            if self.args.save_strategy == SaveStrategy.BEST:
                self.control.should_save = is_new_best_metric
        if self.control.should_save:
            if int(epoch) in [2, 3, 4, 5]:
                self._save_checkpoint(model, trial)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)

                save_epoch_dir = os.path.join(self.args.output_dir, f"epoch-{int(epoch) + 1}")
                os.makedirs(save_epoch_dir, exist_ok=True)
                self.model.save_pretrained(save_epoch_dir)
                if self.tokenizer is not None:
                    self.tokenizer.save_pretrained(save_epoch_dir)
                print(f"✅ [Save] model of epoch-{int(epoch)} saved to {save_epoch_dir}")


def chunked_entropy_from_logits(chunk_logits, batch_size=None):
    """
    Compute entropy from logits in a memory-efficient manner by introducing a batch_size parameter.

    Args:
        chunk_logits (torch.Tensor): Logits tensor of shape (total_samples, num_classes).
        batch_size (int): Number of samples to process per batch.

    Returns:
        torch.Tensor: Entropy tensor of shape (total_samples,).
    """
    total_samples, num_classes = chunk_logits.shape
    entropy_list = []
    if batch_size is None:
        batch_size = total_samples

    # Process logits in batches
    for start_idx in range(0, total_samples, batch_size):
        end_idx = min(start_idx + batch_size, total_samples)
        logits_batch = chunk_logits[start_idx:end_idx]  # Get a batch of logits

        # Compute logsumexp for the current batch
        logsumexp_batch = torch.logsumexp(logits_batch, dim=-1, keepdim=False)  # Shape: (batch_size,)
        # Compute probabilities in log-space without computing softmax
        normalized_logits = logits_batch - logsumexp_batch.unsqueeze(-1)  # Shape: (batch_size, num_classes)
        exp_normalized_logits = torch.exp(normalized_logits)  # Shape: (batch_size, num_classes)
        # Compute entropy for the batch
        entropy_batch = logsumexp_batch - (logits_batch * exp_normalized_logits).sum(dim=-1)  # Shape: (batch_size,)

        entropy_list.append(entropy_batch)  # Store entropy for the current batch

    # Concatenate results from all batches
    if len(entropy_list) > 0:
        return torch.cat(entropy_list, dim=0)
    else:
        return torch.tensor(0.0)
