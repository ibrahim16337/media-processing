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

DEFAULT_SYSTEM_MESSAGE = (
    "ہمیشہ ان پٹ کی زبان کا احترام کریں اور اگر ان پٹ اردو میں ہو تو اردو میں واضح، "
    "درست اور باوقار انداز میں جواب دیں۔\n\n"
    "Return only valid JSON. Do not use markdown. Do not add explanations. "
    "The JSON object must contain exactly these keys: title, description, tags, hashtags."
)


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
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Build the Ollama /api/chat payload.

    Important for Qwen3-style thinking models:
    - think=False makes Ollama return the final answer in message.content.
    - format="json" pushes the model toward strict JSON output.
    - /no_think is added as an extra instruction for Qwen-style models.
    """
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
        "think": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": f"{system_message}\n\n/no_think",
            },
            {
                "role": "user",
                "content": f"{user_prompt}\n\n/no_think\nReturn only strict JSON.",
            },
        ],
        "options": options,
    }


def extract_ollama_content(response: dict[str, Any]) -> str:
    """
    Extract assistant final content from Ollama responses.

    For /api/chat, the normal path is response["message"]["content"].
    For older /api/generate-style responses, response["response"] is supported.
    """
    if "message" in response and isinstance(response["message"], dict):
        content = response["message"].get("content")
        if isinstance(content, str):
            return content.strip()

    if "response" in response and isinstance(response["response"], str):
        return response["response"].strip()

    return ""


def describe_empty_ollama_response(response: dict[str, Any]) -> str:
    message = response.get("message", {})
    thinking = ""

    if isinstance(message, dict):
        raw_thinking = message.get("thinking", "")
        if isinstance(raw_thinking, str):
            thinking = raw_thinking.strip()

    done_reason = response.get("done_reason", "")
    model = response.get("model", "")

    details = [
        "Ollama returned empty message.content.",
        "The app expected JSON metadata but received no final answer text.",
    ]

    if model:
        details.append(f"Model: {model}")

    if done_reason:
        details.append(f"Done reason: {done_reason}")

    if thinking:
        details.append(
            "The response contained thinking text but no final content. "
            "This usually means the model was in thinking mode."
        )

    details.append(
        "Fix attempted by this client: think=false, format=json, and /no_think are sent in the payload."
    )

    return " ".join(details)


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
    system_message: str = DEFAULT_SYSTEM_MESSAGE,
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
    last_response: dict[str, Any] = {}

    for attempt in range(retries + 1):
        try:
            response = call_ollama_chat(
                base_url=base_url,
                payload=payload,
                timeout=timeout,
            )
            last_response = response

            content = extract_ollama_content(response)

            if content:
                return {
                    "ok": True,
                    "content": content,
                    "raw_response": response,
                    "error": "",
                    "attempt": attempt + 1,
                }

            last_error = RuntimeError(describe_empty_ollama_response(response))

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
        "raw_response": last_response,
        "error": str(last_error) if last_error else "Unknown Ollama error",
        "attempt": retries + 1,
    }
