import os
import json
import textwrap
import urllib.request
import urllib.error
import sys
from pathlib import Path
import traceback
import warnings

# suppress noisy warnings for a clean console demo
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*deprecated.*")


def make_zero_shot_prompt(context: str, question: str) -> str:
    """Create a zero-shot prompt using the same structure as `medibot.py`.

    Returns a single string ready to send to a text-generation model.
    """
    prompt = textwrap.dedent(f"""
    Use the pieces of information provided in the context to answer the user's question.
    If you don't know the answer, say you don't know and don't make up an answer.
    Do not provide anything outside the given context.

    Context: {context}
    Question: {question}

    Start the answer directly. No small talk.
    """
    ).strip()
    return prompt


def call_hf_inference(model: str, token: str, prompt: str, max_new_tokens: int = 150) -> str:
    """Call the Hugging Face Inference API (no external deps) and return the generated text.

    Requires environment network access and a valid `token`. If the call fails an exception is raised.
    """
    url = f"https://api-inference.huggingface.co/models/{model}"
    data = json.dumps({
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": 0.0},
        # request raw text completion when possible
        "options": {"wait_for_model": True},
    }).encode("utf-8")

    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            resp_data = resp.read().decode("utf-8")
            parsed = json.loads(resp_data)
            # HF sometimes returns a list of dicts with 'generated_text' or just a string
            if isinstance(parsed, list) and parsed:
                first = parsed[0]
                if isinstance(first, dict) and "generated_text" in first:
                    return first["generated_text"].strip()
                # some models return [{"generated_text": ...}]
                return json.dumps(parsed)
            if isinstance(parsed, dict) and "generated_text" in parsed:
                return parsed["generated_text"].strip()
            # fallback to string representation
            return str(parsed)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HF inference HTTP error: {e.code} {e.reason} - {e.read().decode('utf-8')}")
    except Exception as e:
        raise RuntimeError(f"HF inference failed: {e}")


def load_dotenv_if_present(dotenv_path: str = ".env"):
    """Lightweight .env loader: read key=value lines and set into os.environ if not already set.

    This avoids adding extra dependencies and keeps secrets out of printed output.
    """
    p = Path(dotenv_path)
    # also try the directory where this script lives (helps when running from parent folder)
    script_dir = Path(__file__).resolve().parent
    candidates = [p, script_dir / dotenv_path]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if not found:
        return
    try:
        for line in found.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            # don't print secrets; only set into environment when missing
            if key and val and key not in os.environ:
                os.environ[key] = val
    except Exception:
        # silence errors reading .env to keep demo robust
        return


def naive_local_answer(context: str, question: str) -> str:
    """A deterministic, dependency-free fallback that attempts to answer using context.

    This is NOT a real LLM — it only demonstrates the zero-shot prompt formatting and
    shows a safe fallback when no API key is available.
    """
    ctx = context.lower()
    q = question.lower()
    # simple heuristic: return any sentence in context containing a keyword from the question
    for sent in context.split("."):
        s = sent.strip()
        if not s:
            continue
        for token in q.split():
            if len(token) > 3 and token in s.lower():
                return s.strip() + ". (extracted from context - fallback)"
    return "I don't know based on the provided context. (fallback)"


def call_groq_model(model_name: str, api_key: str, prompt: str) -> str:
    """Attempt to call Groq via langchain_groq.ChatGroq. Returns text or raises RuntimeError.

    This function intentionally keeps errors opaque about secrets and instead returns
    helpful install/permission guidance.
    """
    try:
        from langchain_groq import ChatGroq
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: langchain_groq is not installed. Install with `pip install langchain-groq`."
        )

    try:
        llm = ChatGroq(model_name=model_name, temperature=0.0, groq_api_key=api_key)
    except Exception as e:
        raise RuntimeError("Failed to instantiate ChatGroq. Check GROQ_API_KEY and model name.")

    # Try callable interface
    last_err = None

    # 1) Simple string or object call
    try:
        res = llm(prompt)
        # If object with 'content' attribute
        if hasattr(res, "content"):
            return str(getattr(res, "content")).strip()
        if isinstance(res, str):
            return res.strip()
        if isinstance(res, dict):
            for k in ("content", "text", "result", "output", "generated_text"):
                if k in res:
                    return str(res[k]).strip()
            return json.dumps(res)
        if isinstance(res, list):
            return json.dumps(res)
    except Exception as e:
        last_err = e

    # 2) Try invoke with input key if available
    try:
        if hasattr(llm, "invoke"):
            res = llm.invoke({"input": prompt})
            if hasattr(res, "content"):
                return str(getattr(res, "content")).strip()
            if isinstance(res, str):
                return res.strip()
            if isinstance(res, dict):
                for k in ("content", "text", "result", "output", "generated_text"):
                    if k in res:
                        return str(res[k]).strip()
            return str(res)
    except Exception as e:
        last_err = e

    # 3) Try chat-style messages using HumanMessage (LangChain chat models expect message objects)
    try:
        HM = None
        try:
            from langchain.schema import HumanMessage
            HM = HumanMessage
        except Exception:
            try:
                from langchain_core.schema import HumanMessage
                HM = HumanMessage
            except Exception:
                HM = None

        if HM is not None:
            try:
                msg = HM(content=prompt)
                res = llm([msg])
                if hasattr(res, "content"):
                    return str(getattr(res, "content")).strip()
                if isinstance(res, str):
                    return res.strip()
                if isinstance(res, dict):
                    for k in ("content", "text", "result", "output", "generated_text"):
                        if k in res:
                            return str(res[k]).strip()
                return str(res)
            except Exception as e:
                last_err = e
    except Exception:
        pass

    # 4) Try generate() (LangChain-style)
    try:
        gen = llm.generate([prompt])
        if hasattr(gen, "generations"):
            gens = gen.generations
            if gens and gens[0]:
                first = gens[0][0]
                text = getattr(first, "text", None) or getattr(first, "content", None) or str(first)
                return text.strip()
        return str(gen)
    except Exception as e:
        last_err = e

    raise RuntimeError(f"All Groq invocation attempts failed. Last error: {last_err}")


def main():
    # load .env so HF_TOKEN/HUGGINGFACE_HUB_TOKEN present in environment if file exists
    load_dotenv_if_present()
    # Example context and question — adjust as needed
    context = (
        "Migraine is a neurological condition characterized by recurrent headaches. "
        "Symptoms may include nausea, sensitivity to light and sound, and visual disturbances."
    )
    question = "What are common symptoms of a migraine?"

    prompt = make_zero_shot_prompt(context, question)

    # For a clean demo output we do not print the prompt or any headers.

    hf_token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    # If user provided a preferred model via env, prefer it. Otherwise try a short list of commonly hosted models.
    preferred = os.environ.get("HF_MODEL")
    candidate_models = [preferred] if preferred else []
    # common small hosted text-generation models
    candidate_models += [
        "gpt2",
        "distilgpt2",
        "sshleifer/tiny-gpt2",
        "google/flan-t5-small",
        "bigscience/bloom-1b1",
    ]

    def try_models(token, prompt, candidates):
        last_err = None
        for m in [c for c in candidates if c]:
            try:
                print(f"Trying model: {m}")
                out = call_hf_inference(m, token, prompt)
                return m, out
            except Exception as e:
                last_err = e
                print(f"Model {m} failed: {e}")
        raise RuntimeError(f"All candidate models failed. Last error: {last_err}")

    # Prefer Groq if key is present
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        groq_model = os.environ.get("GROQ_MODEL", "meta-llama/llama-4-maverick-17b-128e-instruct")
        try:
            out = call_groq_model(groq_model, groq_key, prompt)
            print(out)
            return
        except Exception:
            pass

    if hf_token:
        try:
            model_used, out = try_models(hf_token, prompt, candidate_models)
            print(out)
            return
        except Exception:
            pass

    print(naive_local_answer(context, question))


if __name__ == "__main__":
    main()
