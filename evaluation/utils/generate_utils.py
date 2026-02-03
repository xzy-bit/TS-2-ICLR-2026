import torch

def _apply_top_k_top_p(logits,k,p) -> torch.Tensor:
        logits_sort, logits_idx = logits.sort(dim=-1, descending=False)

        top_k_threshold = logits_sort[:, -k].unsqueeze(1)  # shape: [B, 1]
        top_k_mask = logits_sort < top_k_threshold
        logits_sort = logits_sort.masked_fill(top_k_mask, -float("inf"))
        # Apply top-p.
        probs_sort = torch.softmax(logits_sort, dim=-1)
        probs_sum = probs_sort.cumsum(dim=-1)
        top_p_mask = probs_sum <= (1 - p)
        top_p_mask[:, -1] = False  # Ensure at least one token is kept
        logits_sort = logits_sort.masked_fill(top_p_mask, -float("inf"))

        logits_filtered = torch.empty_like(logits_sort).scatter(dim=-1, index=logits_idx, src=logits_sort)
        return logits_filtered
