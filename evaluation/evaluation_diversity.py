#################
# This code is modified from https://github.com/facebookresearch/rlfh-gen-div
#################
import os
from dataclasses import dataclass, field
import json
from pprint import pprint

import torch
import numpy as np
import sentence_transformers
from tqdm import tqdm

# from sklearn.metrics.pairwise import cosine_similarity
from transformers import set_seed, HfArgumentParser, AutoTokenizer

from nltk.util import ngrams
from nltk import word_tokenize
from collections import Counter

import sacrebleu

@dataclass
class AllArguments:
    response_path: str = field(
        default="./results/responses", metadata={"help": "Response path (json file)."}
    )

    tokenizer_path: str = field(default=None)
    detokenizer_path: str = field(default=None)


class SentBertSimilarity:
    def __init__(self):

        self.model_name = "bert-large-nli-stsb-mean-tokens"  # FIXME - hard coded
        self.model = sentence_transformers.SentenceTransformer(self.model_name)
        if torch.cuda.is_available():
            self.model.to(torch.device("cuda"))

    # @functools.cache
    def embed(self, sentence):
        return self.model.encode(sentence)

    # @functools.cache
    def sent_bert_cosine_similarity(self, resps_1, resps_2):
        embeds_1 = self.model.encode(
            resps_1, batch_size=1024, convert_to_tensor=True, show_progress_bar=False
        )
        embeds_2 = self.model.encode(
            resps_2, batch_size=1024, convert_to_tensor=True, show_progress_bar=False
        )

        if torch.cuda.is_available():
            embeds_1 = embeds_1.to(torch.device("cuda"))
            embeds_2 = embeds_2.to(torch.device("cuda"))

        dot_product = (embeds_1 * embeds_2).sum(dim=1)

        # Calculate cosine similarity
        cosine_similarity = dot_product / (embeds_1.norm(dim=1) * embeds_2.norm(dim=1))

        return cosine_similarity.detach().cpu().numpy()

    def __call__(self, resp_a, resp_b):
        return self.sent_bert_cosine_similarity(resp_a, resp_b)


class SentBertDiversity:
    """
    Implements the diversity to similarity reduction specified on section 5 in the paper
    (https://arxiv.org/pdf/2004.02990.pdf)
    for any similarity metric.

    config:
        shared with the original similarity metric.

    usage:
        metric = Similarity2DiversityMetric(config, SimilarityMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see CosineSimilarity2Diversity
    """

    def __init__(self):
        self.similarity_metric = SentBertSimilarity()

    def __call__(self, response_set):
        similarity_list = []
        for i in tqdm(range(len(response_set))):
            for j in range(i):
                similarity_list.append(
                    self.similarity_metric(response_set[i], response_set[j])
                )
        diversity_score = 1 - np.mean(similarity_list)
        return diversity_score


class AveragedNgramDiversityMetric:
    """
    Calculates the mean values of an n-gram based diversity metric in range n in [n_min, n_max].

    config:
        shared with the original n-gram metric.
        n_min(int) > 0 - Specify the lowest n-gram value to be averaged
        n_max(int) > 0 - Specify the highest n-gram value to be averaged

    usage:
        metric = AveragedNgramDiversityMetric(config, NgramMetricClassName)
        metric(response_set)

    inheritance guidelines:
        implement __init__ only

    inheritance example:
        see AveragedDistinctNgrams
    """

    def __init__(self, n_min, n_max):
        # add n field
        self.n_min = n_min
        self.n_max = n_max

    def __call__(self, response_set):
        ngrams_results = []
        num_set = len(response_set)
        for i in range(len(response_set[0])):
            for n in range(self.n_min, self.n_max + 1):
                result = self.calculate_distinct_n(
                    [response_set[j][i] for j in range(num_set)], n
                )
                ngrams_results.append(result)
        return np.mean(ngrams_results)

    def calculate_distinct_n(self, responses, n):
        all_ngrams = []
        for response in responses:
            tokens = word_tokenize(response)
            response_ngrams = list(ngrams(tokens, n))
            all_ngrams.extend(response_ngrams)
        unique_ngrams = len(set(all_ngrams))
        total_ngrams = len(all_ngrams)

        return unique_ngrams / total_ngrams if total_ngrams > 0 else 0


class SelfBLEUMetric:
    def __call__(self, response_set):
        """Calculate the average Self-BLEU score for a list of texts."""
        bleu_scores = []
        k = len(response_set)
        for i in range(len(response_set[0])):
            texts = [response_set[j][i] for j in range(k)]
            bleu_scores.append(self.calculate_bleu_score(texts))

        return np.mean(bleu_scores)

    def calculate_bleu_score(self, texts):
        bleu_scores = []
        for i in range(len(texts)):
            # Treat the current text as the hypothesis
            hypothesis = texts[i]
            # Treat all other texts as references
            references = texts[:i] + texts[i + 1 :]

            if references:  # Ensure there are references to compare against
                bleu_score = sacrebleu.corpus_bleu([hypothesis], [references])
                bleu_scores.append(bleu_score.score)

        # Compute the average BLEU score
        average_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
        return average_bleu


def main():
    parser = HfArgumentParser((AllArguments,))
    (args,) = parser.parse_args_into_dataclasses()
    pprint(args.__dict__)

    if os.path.exists(args.response_path.replace(".json", "-cleaned.json")):
        args.response_path = args.response_path.replace(".json", "-cleaned.json")

    if args.response_path.endswith("-cleaned.json"):
        response_set = json.load(open(args.response_path, "r"))
    else:
        data = json.load(open(args.response_path, "r"))

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if args.detokenizer_path is not None:
            detokenizer = AutoTokenizer.from_pretrained(args.detokenizer_path)
        else:
            detokenizer = None

        response_set = []
        for i in tqdm(range(len(data))):
            n = len(data[i]["answer"])
            if len(response_set) == 0:
                response_set = [[] for _ in range(n)]
            else:
                assert len(response_set) == n
            for j in range(n):
                x = data[i]
                if detokenizer:
                    prompt_str = (
                        detokenizer.decode(
                            detokenizer.encode(x["prompt"]), skip_special_tokens=True
                        )
                        .replace("user\n\n", "")
                        .replace("assistant\n\n", "")
                    )
                else:
                    prompt_str = x["prompt"]
                if detokenizer:
                    # ans_str = detokenizer.decode(
                    #     detokenizer.encode(data[i]["answer"][j]), skip_special_tokens=True
                    # )
                    ans_str = data[i]["answer"][j].replace("<|eot_id|>", "")
                else:
                    ans_str = data[i]["answer"][j]
                chat = [
                    {
                        "role": "user",
                        "content": prompt_str,
                    },
                    {"role": "assistant", "content": ans_str},
                ]
                res = tokenizer.apply_chat_template(chat, tokenize=False)
                response_set[j].append(res)
        json.dump(
            response_set,
            open(args.response_path.replace(".json", "-cleaned.json"), "w"),
            indent=2,
        )

        response_set = json.load(
            open(args.response_path.replace(".json", "-cleaned.json"), "r")
        )
        print("Finished Data Preparation.")

    evaluation_results = {
        "sentbert_diversity_score": None,
        "bleu_diversity_score": None,
        "averaged_ngram_diversity_score": None,
    }

    print("Calculating N-gram diversity score...")
    metric = AveragedNgramDiversityMetric(n_min=1, n_max=3)
    diversity_score = metric(response_set)
    evaluation_results["averaged_ngram_diversity_score"] = np.round(
        diversity_score * 100, 2
    )
    print("N-gram diversity score: {}".format(diversity_score))

    print("Calculating BLEU similarity score...")
    metric = SelfBLEUMetric()
    similarity_score = metric(response_set)
    evaluation_results["bleu_diversity_score"] = np.round(100 - similarity_score, 2)
    print("BLEU similarity score: {}".format(100 - similarity_score))

    print("Calculating Bert diversity score...")
    metric = SentBertDiversity()
    diversity_score = metric(response_set)
    evaluation_results["sentbert_diversity_score"] = np.round(diversity_score * 100, 2)
    print("Bert diversity score: {}".format(diversity_score))

    pprint(evaluation_results)


if __name__ == "__main__":
    main()
