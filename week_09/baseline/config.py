"""Model identity, prompts and saved-format version."""
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = (
    "Answer the factual question using your own knowledge. "
    "You may reason briefly, then give a concise final answer."
)

SCHEMA = "week09-evaluation-v2"

# AgentFlow Appendix E.3; JSON output and alternative references match our interface.
JUDGE_PROMPT = """Task: Determine if the Model Response is equivalent to the Ground Truth.

Instructions:
1. Extract: Isolate the final answer from the Model Response, ignoring all reasoning
   steps. Look specifically for the marked final answer or the concluding statement.
2. Normalize & Compare: Assess equivalence after normalization.
3. Mathematical Answers: Must be mathematically identical (e.g., 1/2 is equivalent to 0.5).
4. Numerical/Textual Answers: Ignore formatting (commas, spaces), case sensitivity,
   and extraneous units/currency (e.g., "1,000" == "1000", "Paris" == "PARIS").
5. Multiple Choice Questions (MCQ): The answer must match either the correct option's
   content (e.g., "Paris") or its identifier (e.g., "A" or "1st").
6. Verdict: Return true only if the normalized answers are semantically or
   mathematically equivalent.

Inputs are supplied as JSON: question is the Question, candidate_response is the
Model Response, and reference_answers contains alternative acceptable Ground Truth
answers. Equivalence to any one reference answer is sufficient.

Output Format: Return only a JSON object with exactly two fields:
"analysis": a brief analysis of the comparison;
"correct": true or false (a JSON boolean).
Do not include any extra text outside the JSON object."""
