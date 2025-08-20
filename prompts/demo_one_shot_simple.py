import os
import json
import textwrap
import urllib.request
import urllib.error
from pathlib import Path


def load_dotenv_if_present(dotenv_path: str = ".env"):
    p = Path(dotenv_path)
    script_dir = Path(__file__).resolve().parent
    candidates = [p, script_dir / dotenv_path]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if not found:
        return
    for line in found.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and val and key not in os.environ:
            os.environ[key] = val


def make_one_shot_prompt(example_ctx: str, example_q: str, example_a: str, ctx: str, q: str) -> str:
    tpl = textwrap.dedent(f"""
    Example:
    Context: {example_ctx}
    Question: {example_q}
    Answer: {example_a}

    Now answer the following using only the context provided.
    Context: {ctx}
    Question: {q}
    Answer:
    """
    )
    return tpl.strip()


def call_hf_inference(model: str, token: str, prompt: str) -> str:
    url = f"https://api-inference.huggingface.co/models/{model}"
    data = json.dumps({
        "inputs": prompt,
        "parameters": {"temperature": 0.0, "max_new_tokens": 150},
        "options": {"wait_for_model": True},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp_data = resp.read().decode("utf-8")
        parsed = json.loads(resp_data)
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "generated_text" in parsed[0]:
            return parsed[0]["generated_text"].strip()
        if isinstance(parsed, dict) and "generated_text" in parsed:
            return parsed["generated_text"].strip()
        return str(parsed)


def naive_fallback_answer(context: str, question: str) -> str:
    for sent in context.split('.'):
        s = sent.strip()
        if not s:
            continue
        for token in question.split():
            if len(token) > 3 and token.lower() in s.lower():
                return s + '.'
    return "I don't know based on the provided context."


def call_groq_if_available(model_name: str, api_key: str, prompt: str):
    try:
        from langchain_groq import ChatGroq
    except Exception:
        return None, "missing_groq"
    try:
        llm = ChatGroq(model_name=model_name, temperature=0.0, groq_api_key=api_key)
    except Exception:
        return None, "init_failed"
    try:
        res = llm(prompt)
        if hasattr(res, "content"):
            return str(getattr(res, "content")).strip(), None
        if isinstance(res, dict):
            for k in ("content", "text", "result", "output", "generated_text"):
                if k in res:
                    return str(res[k]).strip(), None
        if isinstance(res, str):
            return res.strip(), None
        return str(res), None
    except Exception:
        return None, "call_failed"


def main():
    load_dotenv_if_present()

    # One-shot example (small and clear)
    example_ctx = "Asthma is a chronic respiratory condition that causes inflammation and narrowing of the airways."
    example_q = "What are common triggers of asthma?"
    example_a = "Common triggers include allergens, cold air, exercise, and respiratory infections."

# Real context & question (changed to hypertension)
    context = (
    "Hypertension, or high blood pressure, is a condition in which the force of the blood against artery walls is too high. "
    "It often has no symptoms but can lead to serious complications such as heart disease and stroke."
)
    question = "What are potential complications of hypertension?"

    prompt = make_one_shot_prompt(example_ctx, example_q, example_a, context, question)

    # Prefer Groq if key present
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        groq_model = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
        out, err = call_groq_if_available(groq_model, groq_key, prompt)
        if out:
            print(out)
            return

    # Try Hugging Face if token present
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        # try model from env or a small candidate
        hf_model = os.environ.get("HF_MODEL", "google/flan-t5-small")
        try:
            ans = call_hf_inference(hf_model, hf_token, prompt)
            print(ans)
            return
        except Exception:
            pass

    # Fallback
    print(naive_fallback_answer(context, question))


if __name__ == "__main__":
    main()
