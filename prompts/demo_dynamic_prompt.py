"""
Demo: Dynamic Prompting + Token-by-token (streaming) output
This script builds a prompt from hardcoded context and question, selects a hardcoded answer,
then streams the answer token-by-token to simulate token-level generation from an LLM.
No external APIs or tokens required.
"""

import re
import time
import sys
from typing import List


def make_dynamic_prompt(context: str, question: str) -> str:
    return f"Answer the question using the context.\nContext: {context}\nQuestion: {question}\n"


def simple_tokenize(text: str) -> List[str]:
    # Split into words and punctuation tokens, preserving punctuation as separate tokens
    return re.findall(r"\w+|[^\s\w]", text)


def mock_model_generate(prompt: str) -> str:
    # Very small hardcoded "model" that returns an answer based on keywords in the prompt.
    p = prompt.lower()
    if "asthma" in p:
        return (
            "Common triggers include allergens such as pollen and dust mites, cold air, exercise, "
            "respiratory infections, smoke, and certain strong odors."
        )
    if "diabetes" in p:
        return (
            "Symptoms of diabetes can include frequent urination, excessive thirst, unexplained weight loss, "
            "fatigue, blurred vision, and slow-healing sores."
        )
    # Default fallback answer
    return "I don't know based on the provided context."


def stream_tokens(text: str, delay: float = 0.06):
    tokens = simple_tokenize(text)
    out = []
    for i, token in enumerate(tokens):
        # Decide whether to prepend a space. If token is alphanumeric and previous out token ended with alnum, add space.
        if out and re.match(r"\w", token) and re.match(r"\w", out[-1][-1]):
            sys.stdout.write(" ")
        sys.stdout.write(token)
        sys.stdout.flush()
        out.append(token)
        time.sleep(delay)
    sys.stdout.write("\n")
    return "".join([t if re.match(r"[^\w]", t) else (" " + t) for t in out]).strip()


def main():
    # Hardcoded context and question (you can change these)
    context = (
        "Asthma is a chronic respiratory condition that causes inflammation and narrowing of the airways. "
        "Common triggers include allergens, cold air, exercise, and respiratory infections."
    )
    question = "What are common triggers of asthma?"

    prompt = make_dynamic_prompt(context, question)
    print("--- Dynamic Prompt ---")
    print(prompt)
    print("--- Token-by-token (simulated) generation ---")

    answer = mock_model_generate(prompt)
    # Stream the answer tokens to stdout
    final = stream_tokens(answer, delay=0.06)

    print("\n--- Final Answer (reconstructed) ---")
    print(final)


if __name__ == "__main__":
    main()