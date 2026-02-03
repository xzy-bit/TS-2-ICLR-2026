import re
from fraction import Fraction
import numpy as np


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata

        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    completion_new = completion.replace("the answer is", "The answer is")
    text = completion_new.split("The answer is")
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract_ans)
        if match:
            if "/" in match.group():
                denominator = match.group().split("/")[1]
                numerator = match.group().split("/")[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == "0":
                        return round(float(numerator.replace(",", "")))
                    else:
                        frac = Fraction(match.group().replace(",", ""))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(",", "")) == float("inf"):
                    return None
                return round(float(match.group().replace(",", "")))
        else:
            return None
    else:
        return None

def evaluation_answers(
    true_answers, predicted_answers, eos_tokens=["<|eot_id|>"], print_results=False
):
    assert len(true_answers) == len(
        predicted_answers
    ), "true answers (size={}) != predicted answers (size={})".format(
        len(true_answers), len(predicted_answers)
    )
    predictions = []
    skipped_indices = []
    for i in range(len(predicted_answers)):
        final_answer = extract_answer_number(predicted_answers[i])
        if final_answer is not None:
            predictions.append(final_answer)
        else:
            skipped_indices.append(i)

    references = []
    for i in range(len(true_answers)):
        if i not in skipped_indices:
            final_answer = extract_answer_number(true_answers[i])
            references.append(final_answer)

    assert len(predictions) == len(
        references
    ), "predictions (size={}) != references (size={})".format(
        len(predictions), len(references)
    )

    evaluations = []
    c = 0
    for i in range(len(true_answers)):
        if i in skipped_indices:
            evaluations.append((None, None, None))
        else:
            evaluations.append(
                (references[c], predictions[c], references[c] == predictions[c])
            )
            c += 1
    if print_results:
        print(
            "There are {}/{} matched results from the given strings. Acc: {:.4f}".format(
                len(predictions),
                len(true_answers),
                np.mean([x[-1] for x in evaluations if x[-1] is not None]),
            )
        )
    return evaluations
