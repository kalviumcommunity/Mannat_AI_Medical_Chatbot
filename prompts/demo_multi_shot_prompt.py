"""
Demo: Multi-shot (few-shot) Prompting + simulated token streaming

This script demonstrates how to build a multi-shot prompt by combining several
example (context, question, answer) pairs with a new context/question. It
includes a tiny example selector (keyword overlap), dynamic insertion of
examples, and a mock model that generates an answer based on matching examples.

No external API or tokens required — everything is hardcoded for demo purposes.
"""

import re
from typing import List, Dict, Tuple


EXAMPLES: List[Dict[str, str]] = [
	{
		"context": "Asthma is a chronic respiratory condition causing airway inflammation. Common triggers are pollen, dust mites, cold air, exercise, and infections.",
		"question": "What triggers asthma attacks?",
		"answer": "Common triggers include allergens (pollen, dust mites), cold air, exercise, respiratory infections, smoke, and strong odors.",
	},
	{
		"context": "Diabetes is a metabolic disorder where the body cannot regulate blood sugar properly. Symptoms include frequent urination, excessive thirst, and fatigue.",
		"question": "What are symptoms of diabetes?",
		"answer": "Symptoms include frequent urination, excessive thirst, unexplained weight loss, fatigue, blurred vision, and slow-healing wounds.",
	},
	{
		"context": "Hypertension (high blood pressure) often has no symptoms but increases risk of heart disease and stroke. Lifestyle factors include salt intake and lack of exercise.",
		"question": "What are complications of hypertension?",
		"answer": "Complications can include heart disease, stroke, kidney damage, and vision problems.",
	},
]



def score_example(example: Dict[str, str], ctx: str, q: str) -> int:
	"""Return a simple overlap score between example (context+question) and the new ctx/q."""
	text = (example["context"] + " " + example["question"]).lower()
	tokens = set(re.findall(r"\w+", text))
	target = set(re.findall(r"\w+", (ctx + " " + q).lower()))
	return len(tokens & target)


def select_examples(candidates: List[Dict[str, str]], ctx: str, q: str, k: int = 2) -> List[Dict[str, str]]:
	scored: List[Tuple[int, Dict[str, str]]] = [(score_example(e, ctx, q), e) for e in candidates]
	scored.sort(key=lambda t: t[0], reverse=True)
	# Return top-k with positive score; if none match, return the two first examples
	chosen = [e for s, e in scored if s > 0][:k]
	if not chosen:
		return candidates[:k]
	if len(chosen) < k:
		# pad with other examples
		for s, e in scored:
			if e not in chosen:
				chosen.append(e)
			if len(chosen) >= k:
				break
	return chosen


def build_multi_shot_prompt(examples: List[Dict[str, str]], ctx: str, q: str) -> str:
	parts = [
		"You are a helpful medical assistant. Use the examples to follow the style and be concise. Do not invent facts beyond the provided context.",
		"--- Examples ---",
	]
	for ex in examples:
		parts.append(f"Context: {ex['context']}")
		parts.append(f"Question: {ex['question']}")
		parts.append(f"Answer: {ex['answer']}")
		parts.append("")
	parts.append("--- New Query ---")
	parts.append(f"Context: {ctx}")
	parts.append(f"Question: {q}")
	parts.append("Answer:")
	return "\n".join(parts)


def mock_multi_shot_model(prompt: str, examples: List[Dict[str, str]], ctx: str, q: str) -> str:
	"""Very small heuristic 'model': if an example closely matches, reuse its answer (possibly rephrased).
	Otherwise produce a conservative fallback answer extracted from the context."""
	# If any example question shares >50% token overlap with q, return that example's answer
	q_tokens = set(re.findall(r"\w+", q.lower()))
	for ex in examples:
		ex_q_tokens = set(re.findall(r"\w+", ex["question"].lower()))
		if ex_q_tokens and len(q_tokens & ex_q_tokens) / max(1, len(ex_q_tokens)) > 0.5:
			return ex["answer"] + " (based on example)"

	# Otherwise attempt to extract a sentence from ctx that contains a likely answer token
	ctx_sents = re.split(r"(?<=[.!?])\s+", ctx)
	for sent in ctx_sents:
		s = sent.strip()
		if not s:
			continue
		# if the question contains any token >3 chars that appears in the sentence, return sentence
		for tok in q_tokens:
			if len(tok) > 3 and tok in s.lower():
				return s + " (extracted from context - fallback)"

	return "I don't know based on the provided context."


# Token-by-token streaming removed for a simpler demo; answers are printed whole.


def demo_flow(new_ctx: str, new_q: str):
	print("\n=== Multi-shot Prompting Demo ===\n")
	# Step 1: choose relevant examples dynamically
	chosen = select_examples(EXAMPLES, new_ctx, new_q, k=2)
	print("Selected examples (titles):")
	for i, ex in enumerate(chosen, 1):
		print(f" {i}. {ex['question']}")

	# Step 2: build composite prompt
	prompt = build_multi_shot_prompt(chosen, new_ctx, new_q)
	print("\n--- Assembled Prompt ---")
	print(prompt)

	# Step 3: mock model generation and show the final answer
	print("\n--- Simulated generation ---")
	ans = mock_multi_shot_model(prompt, chosen, new_ctx, new_q)
	print("\n--- Final Answer ---")
	print(ans)


def main():
	# Example 1: question similar to asthma example
	ctx1 = (
		"A patient experiences wheeze and breathlessness. Asthma commonly flares after exposure to pollen, dust, or cold air."
	)
	q1 = "What can trigger my asthma symptoms?"
	demo_flow(ctx1, q1)

	# Example 2: a question that relates to diabetes
	ctx2 = (
		"High blood sugar causes increased urination and thirst. If uncontrolled, it can lead to neuropathy and slow wound healing."
	)
	q2 = "What are symptoms of high blood sugar?"
	demo_flow(ctx2, q2)


if __name__ == "__main__":
	main()

