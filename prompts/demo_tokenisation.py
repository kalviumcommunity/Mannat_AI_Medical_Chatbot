"""
Demo: Tokenisation and token-usage logging

This script builds a prompt (from hardcoded context and question), uses a mock
generator to produce an answer, tokenises prompt and answer, prints token lists
and counts, and writes a small JSON summary to `token_usage_summary.json` in
the `prompts/` folder.

It prefers `tiktoken` if installed (to approximate GPT-style tokenisation); if
not available, it falls back to a simple regex-based tokenizer.
"""

import json
import os
import re
from typing import List


def try_import_tiktoken():
    try:
        import tiktoken

        return tiktoken
    except Exception:
        return None


TIKTOKEN = try_import_tiktoken()


def regex_tokenize(text: str) -> List[str]:
    # Very small fallback tokenizer (words and punctuation)
    return re.findall(r"\w+|[^\s\w]", text)


def tok_encode(text: str) -> List[str]:
    if TIKTOKEN is not None:
        # choose cl100k_base if available, otherwise the first encoding
        try:
            enc = TIKTOKEN.get_encoding("cl100k_base")
        except Exception:
            enc_names = list(TIKTOKEN._encodings.keys()) if hasattr(TIKTOKEN, "_encodings") else []
            enc = TIKTOKEN.get_encoding(enc_names[0]) if enc_names else None
        if enc is not None:
            ids = enc.encode(text)
            # return token ids as strings to avoid exposing raw encoding objects
            return [str(i) for i in ids]
    # fallback: return textual tokens
    return regex_tokenize(text)


def make_prompt(ctx: str, q: str) -> str:
    return f"Answer concisely using the context.\nContext: {ctx}\nQuestion: {q}\n"


def mock_generator(prompt: str) -> str:
    p = prompt.lower()
    if "asthma" in p:
        return "Common triggers: pollen, dust mites, cold air, exercise, infections."
    if "diabetes" in p:
        return "Symptoms: frequent urination, excessive thirst, fatigue, blurred vision."
    return "I don't know based on the provided context."


def log_token_usage(prompt: str, answer: str, out_path: str):
    prompt_tokens = tok_encode(prompt)
    answer_tokens = tok_encode(answer)

    summary = {
        "prompt_length_chars": len(prompt),
        "answer_length_chars": len(answer),
        "prompt_token_count": len(prompt_tokens),
        "answer_token_count": len(answer_tokens),
        "prompt_tokens_sample": prompt_tokens[:50],
        "answer_tokens_sample": answer_tokens[:50],
    }

    print("\n--- Token usage summary ---")
    print(f"Prompt chars: {summary['prompt_length_chars']}")
    print(f"Answer chars: {summary['answer_length_chars']}")
    print(f"Prompt tokens: {summary['prompt_token_count']}")
    print(f"Answer tokens: {summary['answer_token_count']}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote token usage summary to: {out_path}")


def demo_one(ctx: str, q: str, out_dir: str):
    prompt = make_prompt(ctx, q)
    print("--- Prompt ---")
    print(prompt)
    ans = mock_generator(prompt)
    print("--- Answer ---")
    print(ans)
    out_path = os.path.join(out_dir, "token_usage_summary.json")
    log_token_usage(prompt, ans, out_path)


def main():
    out_dir = os.path.dirname(__file__)
    # Demo 1
    ctx1 = (
        "Asthma is a chronic lung disease. Triggers include pollen, dust mites, cold air, and exercise."
    )
    q1 = "What can trigger asthma?"
    demo_one(ctx1, q1, out_dir)

    # Demo 2
    ctx2 = (
        "Diabetes often causes increased thirst and frequent urination. It may cause fatigue and vision changes."
    )
    q2 = "What are symptoms of diabetes?"
    demo_one(ctx2, q2, out_dir)


if __name__ == "__main__":
    print("tiktoken available:" , bool(TIKTOKEN))
    main()
