from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from .config import Config

logger = logging.getLogger(__name__)


class OpenAIClient:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            organization=cfg.openai_org,
            project=cfg.openai_project,
            timeout=cfg.request_timeout_s,
        )

    def upload_file(self, path: Path) -> str:
        with path.open("rb") as f:
            file_obj = self.client.files.create(file=f, purpose="user_data")
        return file_obj.id

    def call_json(
        self,
        *,
        instructions: str,
        input_items: list[dict[str, Any]],
        schema: dict[str, Any],
        schema_name: str,
        prompt_cache_key: str,
        max_output_tokens: int,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            raw = ""
            try:
                response = self.client.responses.create(
                    model=self.cfg.model,
                    instructions=instructions,
                    input=input_items,
                    tools=[],
                    parallel_tool_calls=False,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": schema_name,
                            "strict": True,
                            "schema": schema,
                        }
                    },
                    prompt_cache_key=prompt_cache_key,
                )
                raw = self._extract_output_text(response)
                return self._parse_and_validate(raw, schema)
            except (json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                logger.warning("JSON parse/validation failed, attempting repair")
                try:
                    repaired = self.repair_to_schema(
                        raw_text=raw,
                        schema=schema,
                        schema_name=schema_name,
                        prompt_cache_key=prompt_cache_key + ":repair",
                        max_output_tokens=max_output_tokens,
                    )
                    return repaired
                except Exception as repair_exc:  # noqa: BLE001
                    last_error = repair_exc
            except (RateLimitError, APIError, APITimeoutError, APIConnectionError) as exc:
                last_error = exc
                self._backoff_sleep(attempt)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self._backoff_sleep(attempt)
        raise RuntimeError("OpenAI call failed") from last_error

    def repair_to_schema(
        self,
        *,
        raw_text: str,
        schema: dict[str, Any],
        schema_name: str,
        prompt_cache_key: str,
        max_output_tokens: int,
    ) -> dict[str, Any]:
        instructions = (
            "You are a JSON repair utility. Convert the user-provided text into JSON that "
            "conforms to the provided schema. Return ONLY valid JSON. Do not add extra keys."
        )
        input_items = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Schema:"},
                    {"type": "input_text", "text": json.dumps(schema)},
                    {"type": "input_text", "text": "Raw output:"},
                    {"type": "input_text", "text": raw_text or ""},
                ],
            }
        ]
        response = self.client.responses.create(
            model=self.cfg.model,
            instructions=instructions,
            input=input_items,
            tools=[],
            parallel_tool_calls=False,
            temperature=0,
            max_output_tokens=max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                }
            },
            prompt_cache_key=prompt_cache_key,
        )
        raw = self._extract_output_text(response)
        return self._parse_and_validate(raw, schema)

    def _parse_and_validate(self, raw: str, schema: dict[str, Any]) -> dict[str, Any]:
        data = json.loads(raw)
        Draft202012Validator(schema).validate(data)
        return data

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text
        if hasattr(response, "output"):
            for item in response.output:
                if hasattr(item, "content"):
                    for content in item.content:
                        if getattr(content, "type", "") == "output_text":
                            return content.text
        raise ValueError("No output_text in response")

    @staticmethod
    def _backoff_sleep(attempt: int) -> None:
        base = 1.5 * (2 ** attempt)
        jitter = random.uniform(0, 0.25)
        time.sleep(base + jitter)
