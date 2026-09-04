"""Turns a natural-language edit instruction into a structured action.

Tries a real LLM first (Anthropic / OpenAI / Gemini - whichever is
configured with an API key) and asks it to return strict JSON:

    {"operation": "replace_object"|"remove_object"|"replace_text",
     "target_object": str, "replacement_object": str|null,
     "target_text": str|null, "replacement_text": str|null}

If no provider is configured (or the call fails), we fall back to a
deterministic regex/keyword parser so the whole app still works offline -
this keeps the AI integration real without making the assignment depend on
having an API key.
"""
from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from app.config import settings
from app.schemas import Operation, ParsedInstruction
from app.services.coco_classes import best_coco_match

SYSTEM_PROMPT = """You are an instruction parser for a video editing tool.
Given a user's natural-language video edit request, extract a strict JSON
object describing the edit. Do not include any prose, only JSON.

Schema:
{
  "operation": "replace_object" | "remove_object" | "replace_text",
  "target_object": string,        // the object being replaced/removed, short noun phrase
  "replacement_object": string|null,  // what to replace it with (for replace_object)
  "target_text": string|null,     // literal text to replace (for replace_text)
  "replacement_text": string|null // literal replacement text (for replace_text)
}

Examples:
"Replace the Coca-Cola bottle with Pepsi" ->
{"operation":"replace_object","target_object":"Coca-Cola bottle","replacement_object":"Pepsi","target_text":null,"replacement_text":null}

"Remove the bottle from the table" ->
{"operation":"remove_object","target_object":"bottle","replacement_object":null,"target_text":null,"replacement_text":null}

"Replace Coca-Cola with Pepsi" (label/logo text on the object) ->
{"operation":"replace_text","target_object":"label","replacement_object":null,"target_text":"Coca-Cola","replacement_text":"Pepsi"}
"""


async def _call_anthropic(prompt: str) -> Optional[dict]:
    if not settings.anthropic_api_key:
        return None
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": settings.anthropic_model,
                "max_tokens": 300,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []))
        return _extract_json(text)


async def _call_openai(prompt: str) -> Optional[dict]:
    if not settings.openai_api_key:
        return None
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {settings.openai_api_key}"},
            json={
                "model": settings.openai_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        return _extract_json(text)


async def _call_gemini(prompt: str) -> Optional[dict]:
    if not settings.gemini_api_key:
        return None
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{settings.gemini_model}:generateContent?key={settings.gemini_api_key}"
    )
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            url,
            json={
                "contents": [{"parts": [{"text": f"{SYSTEM_PROMPT}\n\nUser request: {prompt}"}]}],
                "generationConfig": {"temperature": 0},
            },
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return _extract_json(text)


def _extract_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _fallback_parse(prompt: str) -> dict:
    """Deterministic regex/keyword parser used when no LLM key is configured."""
    p = prompt.strip()
    low = p.lower()

    m = re.search(r"replace\s+(?:the\s+)?(.+?)\s+with\s+(.+?)[\.\!]?$", low)
    if m:
        target, replacement = m.group(1).strip(), m.group(2).strip()
        # Heuristic: if the target looks like a short brand word already found
        # as a substring of the original (non-lowered) text near quotes, or if
        # it maps to no plausible physical object, treat it as a text swap.
        if _looks_like_object(target):
            return {
                "operation": "replace_object",
                "target_object": target,
                "replacement_object": replacement,
                "target_text": None,
                "replacement_text": None,
            }
        return {
            "operation": "replace_text",
            "target_object": "label",
            "replacement_object": None,
            "target_text": _original_case(p, target),
            "replacement_text": _original_case(p, replacement),
        }

    m = re.search(r"remove\s+(?:the\s+)?(.+?)[\.\!]?$", low)
    if m:
        return {
            "operation": "remove_object",
            "target_object": m.group(1).strip(),
            "replacement_object": None,
            "target_text": None,
            "replacement_text": None,
        }

    # Last resort: assume a generic replace targeting the whole sentence.
    return {
        "operation": "replace_object",
        "target_object": low,
        "replacement_object": None,
        "target_text": None,
        "replacement_text": None,
    }


OBJECT_HINT_WORDS = (
    "bottle", "can", "cup", "glass", "bag", "box", "phone", "laptop", "book",
    "chair", "table", "car", "hat", "shirt", "shoe", "bowl", "plate", "cap",
    "logo", "label",
)


def _looks_like_object(phrase: str) -> bool:
    return any(w in phrase for w in OBJECT_HINT_WORDS)


def _original_case(original: str, lowered_fragment: str) -> str:
    idx = original.lower().find(lowered_fragment)
    if idx == -1:
        return lowered_fragment
    return original[idx : idx + len(lowered_fragment)]


async def parse_instruction(prompt: str) -> ParsedInstruction:
    result: Optional[dict] = None
    parsed_by = "fallback-regex"

    provider = settings.llm_provider.lower()
    try:
        if provider == "anthropic":
            result = await _call_anthropic(prompt)
            parsed_by = f"anthropic:{settings.anthropic_model}"
        elif provider == "openai":
            result = await _call_openai(prompt)
            parsed_by = f"openai:{settings.openai_model}"
        elif provider == "gemini":
            result = await _call_gemini(prompt)
            parsed_by = f"gemini:{settings.gemini_model}"
    except Exception:
        result = None

    if result is None:
        result = _fallback_parse(prompt)
        parsed_by = "fallback-regex"

    target_object = (result.get("target_object") or "").strip()
    coco_class = best_coco_match(target_object) if target_object else None

    return ParsedInstruction(
        operation=Operation(result.get("operation", "replace_object")),
        target_object=target_object,
        replacement_object=result.get("replacement_object"),
        target_text=result.get("target_text"),
        replacement_text=result.get("replacement_text"),
        coco_class=coco_class,
        raw_prompt=prompt,
        parsed_by=parsed_by,
    )
