from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


DEFAULT_MODEL = "qwen3:30b"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_TIMEOUT = 300
DEFAULT_RETRIES = 2
DEFAULT_SLEEP_MS = 200
DEFAULT_TEMPERATURE = 0.2
DEFAULT_NUM_CTX = 8192
DEFAULT_NUM_PREDICT = 1024


def call_ollama_chat(
    base_url: str,
    payload: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/api/chat",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw.decode("utf-8", errors="replace")}


def build_ollama_payload(
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    system_message: str = "ہمیشہ ان پٹ کی زبان کا احترام کریں اور اگر ان پٹ اردو میں ہو تو اردو میں واضح، درست اور باوقار انداز میں جواب دیں۔",
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "temperature": temperature,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
    }

    if seed is not None:
        options["seed"] = seed

    return {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ],
        "options": options,
    }


def extract_ollama_content(response: dict[str, Any]) -> str:
    if "message" in response and isinstance(response["message"], dict):
        content = response["message"].get("content")
        if isinstance(content, str):
            return content.strip()

    if "response" in response and isinstance(response["response"], str):
        return response["response"].strip()

    return json.dumps(response, ensure_ascii=False, indent=2)


def generate_metadata_from_prompt(
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    system_message: str = "ہمیشہ ان پٹ کی زبان کا احترام کریں اور اگر ان پٹ اردو میں ہو تو اردو میں واضح، درست اور باوقار انداز میں جواب دیں۔",
) -> dict[str, Any]:
    payload = build_ollama_payload(
        user_prompt=user_prompt,
        model=model,
        system_message=system_message,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
    )

    last_error: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = call_ollama_chat(
                base_url=base_url,
                payload=payload,
                timeout=timeout,
            )
            content = extract_ollama_content(response)

            return {
                "ok": True,
                "content": content,
                "raw_response": response,
                "error": "",
                "attempt": attempt + 1,
            }

        except urllib.error.HTTPError as e:
            try:
                error_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = str(e)

            last_error = RuntimeError(f"HTTP {e.code}: {error_body}")

        except urllib.error.URLError as e:
            last_error = RuntimeError(f"Connection error: {e}")

        except Exception as e:
            last_error = e

        if attempt < retries:
            time.sleep(1.0 + attempt)

    time.sleep(max(0, sleep_ms / 1000.0))

    return {
        "ok": False,
        "content": "",
        "raw_response": {},
        "error": str(last_error) if last_error else "Unknown Ollama error",
        "attempt": retries + 1,
    }