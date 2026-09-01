#!/usr/bin/env python3
"""Minimal Azure APIM client for one-teacher-at-a-time trace generation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
SECRETS_PATH = ROOT / ".secrets.env"

_TEACHERS = {
    "gpt-5.6-luna": {"provider": "openai", "model": "gpt-5.6-luna"},
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
    },
}


def _parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse simple KEY=VALUE lines without modifying process environment."""
    values: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _load_credentials(path: str | Path | None = None) -> tuple[str, str]:
    values = _parse_env_file(path or SECRETS_PATH)
    endpoint = (values.get("ORD_API_ENDPOINT")
                or values.get("AZURE_APIM_ENDPOINT")
                or values.get("APIM_ENDPOINT"))
    key = (values.get("ORD_API_KEY")
           or values.get("AZURE_APIM_KEY")
           or values.get("APIM_KEY"))
    if not endpoint or not key:
        raise RuntimeError(
            ".secrets.env must define ORD_API_ENDPOINT and ORD_API_KEY "
            "(AZURE_APIM_* and APIM_* aliases are also accepted)."
        )
    return endpoint.rstrip("/"), key


def _load_key_ring(path: str | Path | None = None) -> tuple[str, list[str]]:
    """Endpoint plus one key per quota PROFILE (secondaries share quota
    with their primary, so they are not useful for rotation)."""
    values = _parse_env_file(path or SECRETS_PATH)
    endpoint, first = _load_credentials(path)
    ring = [first]
    for name in ("ORD_API_KEY_P2A", "ORD_API_KEY_P3A"):
        if values.get(name):
            ring.append(values[name])
    return endpoint, ring


def _teacher_spec(teacher: str) -> dict[str, str]:
    try:
        return _TEACHERS[teacher]
    except KeyError as exc:
        supported = ", ".join(sorted(_TEACHERS))
        raise ValueError(f"Unknown teacher {teacher!r}; choose {supported}") from exc


def _build_openai_payload(model: str, messages: list[dict],
                          max_tokens: int, temperature: float | None,
                          logprobs: bool) -> dict:
    payload = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
    }
    if logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = 5
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def _build_anthropic_payload(model: str, messages: list[dict],
                             max_tokens: int,
                             temperature: float | None) -> dict:
    payload = {"model": model, "max_tokens": max_tokens,
               "messages": messages}
    if temperature is not None:
        payload["temperature"] = temperature
    return payload


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in (None, "text"):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    return str(content or "")


def chat(teacher: str, messages: list[dict], max_tokens: int,
         temperature: float | None = None,
         logprobs: bool = False) -> dict:
    """Call one configured teacher and normalize its response."""
    spec = _teacher_spec(teacher)
    endpoint, ring = _load_key_ring()

    response = None
    for key_idx, key in enumerate(ring[_ACTIVE_PROFILE[0]:],
                                  start=_ACTIVE_PROFILE[0]):
        headers = {"api-key": key, "Content-Type": "application/json"}
        if spec["provider"] == "openai":
            url = f"{endpoint}/openai/v1/chat/completions"
            payload = _build_openai_payload(
                spec["model"], messages, max_tokens, temperature, logprobs)
        else:
            url = f"{endpoint}/anthropic/v1/messages"
            headers["anthropic-version"] = "2023-06-01"
            # the anthropic route on this gateway authenticates via x-api-key
            headers["x-api-key"] = key
            payload = _build_anthropic_payload(
                spec["model"], messages, max_tokens, temperature)

        response = requests.post(url, headers=headers, json=payload,
                                 timeout=300)
        if (response.status_code == 400 and logprobs
                and "logprobs" in response.text):
            # model rejects the logprobs parameter entirely -> retry without
            payload.pop("logprobs", None)
            payload.pop("top_logprobs", None)
            response = requests.post(url, headers=headers, json=payload,
                                     timeout=300)
        if (response.status_code == 403
                and "quota" in response.text.lower()
                and key_idx + 1 < len(ring)):
            # this quota profile is exhausted -> rotate to the next one
            _ACTIVE_PROFILE[0] = key_idx + 1
            _note_rotation(key_idx + 1)
            continue
        break
    response.raise_for_status()
    raw = response.json()
    if spec["provider"] == "openai":
        choice = raw.get("choices", [{}])[0]
        text = _content_text(choice.get("message", {}).get("content", ""))
        response_logprobs = choice.get("logprobs")
    else:
        text = _content_text(raw.get("content", []))
        response_logprobs = None
    return {
        "text": text,
        "usage": raw.get("usage"),
        "logprobs": response_logprobs,
        "model_version": raw.get("model", spec["model"]),
    }


# index of the currently active quota profile in the key ring
_ACTIVE_PROFILE = [0]


def _note_rotation(new_index: int) -> None:
    """Record profile rotation so budget monitors can attribute spend."""
    import time
    marker = ROOT / "results/traces-pilot/.profile_rotation"
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "a") as fh:
        fh.write(f"{time.time():.0f} switched_to_profile {new_index}\n")


def _retryable(exc: Exception) -> bool:
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        status = exc.response.status_code
        return status == 429 or 500 <= status < 600
    return False


def _chat_with_retries(teacher: str, prompt: str, max_tokens: int,
                       temperature: float | None, retries: int) -> dict:
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(retries + 1):
        try:
            return chat(teacher, messages, max_tokens, temperature)
        except Exception as exc:
            if attempt >= retries or not _retryable(exc):
                raise
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError("retry loop exited unexpectedly")


def _completed_prompts(path: Path, teacher: str) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_teacher = row.get("teacher")
            if row_teacher is not None and row_teacher != teacher:
                raise RuntimeError(
                    f"{path} already contains teacher {row_teacher!r}; "
                    "use a separate output file for each teacher."
                )
            if isinstance(row.get("prompt"), str):
                completed.add(row["prompt"])
    return completed


def generate_traces(teacher: str, prompts: Iterable[str],
                    out_jsonl: str | Path, max_tokens: int = 1024,
                    temperature: float | None = None, concurrency: int = 4,
                    retries: int = 5) -> list[dict]:
    """Generate resume-safe JSONL traces with at most four concurrent calls."""
    _teacher_spec(teacher)
    if not 1 <= concurrency <= 4:
        raise ValueError("concurrency must be between 1 and 4")
    if retries < 0:
        raise ValueError("retries must be non-negative")

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = _completed_prompts(out_path, teacher)
    pending = []
    for prompt in prompts:
        prompt = str(prompt)
        if prompt not in seen:
            pending.append(prompt)
            seen.add(prompt)

    written: list[dict] = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_chat_with_retries, teacher, prompt, max_tokens,
                        temperature, retries): prompt
            for prompt in pending
        }
        with out_path.open("a", encoding="utf-8") as stream:
            for future in as_completed(futures):
                prompt = futures[future]
                result = future.result()
                row = {
                    "prompt": prompt,
                    "response": result["text"],
                    "teacher": teacher,
                    "model_version": result["model_version"],
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "usage": result["usage"],
                }
                serialized_row = json.dumps(row, ensure_ascii=False) + "\n"
                stream.write(serialized_row)
                stream.flush()
                written.append(row)
    return written


def _probe_prompts(source) -> list[str]:
    if callable(source):
        source = source()
    if isinstance(source, (str, Path)):
        prompts = []
        with Path(source).open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                prompts.append(str(row["prompt"]))
        return prompts
    if isinstance(source, dict):
        source = [item for group in source.values() for item in group]
    return [str(item["prompt"] if isinstance(item, dict) else item)
            for item in source]


def measure_teacher(teacher: str, probes_jsonl_or_builder) -> list[dict]:
    """Run probes and return ungraded responses for downstream scoring."""
    records = []
    for prompt in _probe_prompts(probes_jsonl_or_builder):
        result = chat(teacher, [{"role": "user", "content": prompt}], 1024)
        records.append({
            "prompt": prompt,
            "response": result["text"],
            "teacher": teacher,
            "model_version": result["model_version"],
            "ts": datetime.now(timezone.utc).isoformat(),
            "usage": result["usage"],
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=sorted(_TEACHERS))
    ap.add_argument("--probe", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    result = chat(
        args.teacher,
        [{"role": "user", "content": args.probe}],
        args.max_tokens,
        logprobs=args.teacher == "gpt-5.6-luna",
    )
    print(result["text"])
    print(f"logprobs available: {result['logprobs'] is not None}")


if __name__ == "__main__":
    main()
