"""
Demo: Chain-of-Thought (CoT) Prompting

This script shows how to construct prompts that include example chains-of-thought
(`reasoning steps`) and a target question. It uses hardcoded CoT examples and a
small heuristic 'model' that returns a simulated chain-of-thought followed by a
concise final answer. No external APIs or tokens required.
"""

import re
from typing import List, Dict


EXAMPLES_COT: List[Dict[str, str]] = [
    {
        "context": "A patient reports wheezing and shortness of breath after running outdoors. Triggers include pollen and cold air.",
        "question": "What likely triggered the breathing problem?",
        "cot": (
            "1) Identify symptoms: wheezing and shortness of breath after exercise.\n"
            "2) Match to known triggers: exercise-induced bronchoconstriction and airborne allergens are common.\n"
            "3) Combine facts: symptoms follow running outdoors where pollen/cold air are present.\n"
            "Conclusion: Exercise and environmental allergens (pollen/cold air) likely triggered it."
        ),
        "final": "Exercise and environmental allergens (pollen or cold air) likely triggered the breathing problem."
    },
    {
        "context": "A person has frequent urination, extreme thirst, and unexplained weight loss over several weeks.",
        "question": "What condition is most likely?",
        "cot": (
            "1) Identify key symptoms: polyuria, polydipsia, and weight loss.\n"
            "2) Map symptoms to conditions: these are classic for hyperglycemia/diabetes.\n"
            "3) Likely diagnosis: these systemic metabolic signs most closely match diabetes.\n"
            "Conclusion: Diabetes is the most likely condition."
        ),
        "final": "The most likely condition is diabetes."
    },
]


def build_cot_prompt(examples: List[Dict[str, str]], ctx: str, q: str) -> str:
    parts = [
        "You are a helpful medical assistant. For each example show step-by-step reasoning (short bullets) labeled as 'Thought process' followed by a concise final answer.",
        "--- Chain-of-Thought Examples ---",
    ]
    for ex in examples:
        parts.append(f"Context: {ex['context']}")
        parts.append(f"Question: {ex['question']}")
        parts.append("Thought process:")
        parts.extend(ex['cot'].split('\n'))
        parts.append(f"Final answer: {ex['final']}")
        parts.append("")
    parts.append("--- New Query ---")
    parts.append(f"Context: {ctx}")
    parts.append(f"Question: {q}")
    parts.append("Thought process:")
    parts.append("")
    parts.append("Final answer:")
    return "\n".join(parts)


def mock_cot_model(examples: List[Dict[str, str]], ctx: str, q: str) -> Dict[str, str]:
    """Heuristic model that returns a simulated chain-of-thought and a final answer.
    If the question closely matches an example, reuse that example's chain and answer.
    Otherwise, extract relevant sentence(s) from context and build a brief 3-step reasoning.
    """
    q_tokens = set(re.findall(r"\w+", q.lower()))
    for ex in examples:
        ex_q_tokens = set(re.findall(r"\w+", ex["question"].lower()))
        if ex_q_tokens and len(q_tokens & ex_q_tokens) / max(1, len(ex_q_tokens)) > 0.5:
            return {"cot": ex["cot"], "final": ex["final"]}

    # Build a short CoT from context
    ctx_sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ctx) if s.strip()]
    facts = []
    for s in ctx_sents:
        # collect sentences that share tokens with the question
        for tok in q_tokens:
            if len(tok) > 3 and tok in s.lower():
                facts.append(s)
                break

    if not facts and ctx_sents:
        # fallback: use first sentence as a fact
        facts.append(ctx_sents[0])

    # Keep reasoning short (3 steps)
    cot_lines = []
    cot_lines.append("1) Identify relevant facts: " + ("; ".join(facts) if facts else "none obvious"))
    cot_lines.append("2) Apply medical reasoning: relate the facts to likely causes or diagnoses.")
    cot_lines.append("3) Draw a concise conclusion based on the most relevant fact(s).")

    # Final answer: try to extract a noun phrase or return a conservative fallback
    final = facts[0] if facts else "I don't know based on the provided context."
    # Shorten final to be a concise statement
    if final != "I don't know based on the provided context.":
        # make the sentence into a concise conclusion
        final = final.rstrip('.!?') + '.'

    return {"cot": "\n".join(cot_lines), "final": final}


def demo_flow(ctx: str, q: str):
    print("\n=== Chain-of-Thought Prompting Demo ===\n")
    prompt = build_cot_prompt(EXAMPLES_COT, ctx, q)
    print("--- Assembled Prompt (showing examples + target question) ---")
    print(prompt)

    print("\n--- Simulated Chain-of-Thought Generation ---")
    out = mock_cot_model(EXAMPLES_COT, ctx, q)
    print("Thought process:")
    print(out["cot"])
    print("\nFinal answer:")
    print(out["final"]) 


def main():
    # Demo 1: close match to first example (asthma)
    ctx1 = "A runner develops wheeze and cough when jogging in the park during spring pollen season."
    q1 = "What likely caused the runner's symptoms?"
    demo_flow(ctx1, q1)

    # Demo 2: more generic extraction from context
    ctx2 = "The patient reports increased thirst and urination over several weeks, with recent weight loss."
    q2 = "Which condition should be suspected?"
    demo_flow(ctx2, q2)


if __name__ == '__main__':
    main()
