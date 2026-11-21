"""Answer extraction only; normalized exact match comes from veRL."""
import re


def last_generated_segment(token_ids, response_mask):
    """Never grade an answer copied from a tool response or prompt."""
    if len(token_ids) != len(response_mask):
        raise ValueError("Token/mask lengths differ.")
    if not token_ids or response_mask[-1] != 1:
        return []
    start = len(token_ids) - 1
    while start > 0 and response_mask[start - 1] == 1:
        start -= 1
    return token_ids[start:]


def extract_final_answer(text):
    # A pending tool call is not a final answer, even if its query quotes answer tags.
    if "<tool_call>" in text:
        return None
    matches = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None


def score_final(text, targets):
    from verl.utils.reward_score.search_r1_like_qa_em import em_check
    answer = extract_final_answer(text)
    return {"score": float(em_check(answer, targets)) if answer is not None else 0.0,
            "prediction": answer or "", "has_answer": int(answer is not None)}
