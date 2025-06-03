import numpy as np
from src.prompts import wrap_prompt

def filter_by_attention_score(topk_contents, llm, top_tokens, threshold, max_corruptions, question, choices, dataset):
    removed_count = 0
    contents = topk_contents

    # sorting the passages first time according to the attention scores
    query_prompt = wrap_prompt(question, contents, choices, dataset)
    _, passage_scores, _ = llm.query(query_prompt, top_tokens)
    attention_score_sum = sum(passage_scores)
    sort_normalized_passage_scores = [(x/attention_score_sum) * 100 for x in passage_scores]

    sorted_contents = [c for c, s in sorted(zip(contents, sort_normalized_passage_scores),key=lambda x: x[1])]

    # sorting the passages second time according to the attention scores
    query_prompt = wrap_prompt(question, sorted_contents, choices, dataset)
    _, passage_scores, _ = llm.query(query_prompt, top_tokens)
    attention_score_sum = sum(passage_scores)
    sort_normalized_passage_scores = [(x/attention_score_sum) * 100 for x in passage_scores]
    
    sorted_contents = [c for c, s in sorted(zip(sorted_contents, sort_normalized_passage_scores),key=lambda x: x[1])]

    while removed_count < max_corruptions:
        # using the finally sorted passages to compute the final query prompt used for filtering
        query_prompt = wrap_prompt(question, sorted_contents, choices, dataset)
        _, passage_scores, _ = llm.query(query_prompt, top_tokens)
        attention_score_sum = sum(passage_scores)
        normalized_passage_scores = [(x/attention_score_sum) * 100 for x in passage_scores]

        if np.var(normalized_passage_scores) <= threshold:
            break

        max_score = max(normalized_passage_scores)
        max_index = normalized_passage_scores.index(max_score)
        sorted_contents.pop(max_index)
        contents = sorted_contents
        removed_count +=1
    
    return contents