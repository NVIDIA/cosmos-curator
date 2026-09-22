# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared SDK import guard and stub endpoints for OpenAI / Gemini provider tests.

This is the shared foundation for provider end-to-end tests across both the
video and image pipelines.

:func:`require_provider_module` is the import guard those tests use to acquire a
provider SDK: it makes a missing SDK a hard failure in the CPU unit-test
environment and a skip everywhere else.

On top of that it exposes two complementary stubbing strategies, one per
provider, because the two SDKs are injectable in fundamentally different ways:

* :class:`OpenAIStubServer` — a *real* localhost ``ThreadingHTTPServer`` that
  speaks the OpenAI-compatible REST surface (``POST /v1/chat/completions``,
  ``POST /v1/responses``, ``POST /v1/embeddings`` and ``GET /v1/models``).
  Because the OpenAI clients honour the config ``base_url``, pointing a stage's
  endpoint config at :attr:`OpenAIStubServer.base_url` exercises the real
  ``openai`` SDK end-to-end against canned, SDK-shaped JSON. Each request is
  recorded so tests can assert on exactly what was sent.

* :func:`make_gemini_stub` — an ``AsyncMock`` / ``SimpleNamespace`` fake
  ``genai.Client``. Google GenAI is built via ``genai.Client(api_key=...)`` with
  no ``base_url`` knob, so it cannot be redirected to a local HTTP server
  without provider source changes (explicitly out of scope here). Tests inject
  the fake client instead, matching the existing ``_attach_gemini_async_client``
  pattern but consolidated into one reusable helper.
"""

import contextlib
import importlib
import json
import socketserver
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from cosmos_curator.core.utils.model import pixi_utils

# Provider coverage is required to run in the CPU unit-test job, which activates the pixi ``dev``
# environment. Other pixi environments still import the provider test modules while collecting
# ``pytest -m env`` runs, so a missing provider SDK skips there but fails loudly in ``dev`` rather
# than turning required provider coverage into a silent green skip.
UNIT_TEST_PIXI_ENV_NAME = "dev"

# Logical endpoint keys used to look up / override canned responses.
CHAT_COMPLETIONS = "chat.completions"
RESPONSES = "responses"
EMBEDDINGS = "embeddings"
MODELS = "models"

# Valid keys for OpenAIStubServer.set_response; guards against typos that would
# silently register an unused override while the real route serves the default.
_VALID_ENDPOINTS = frozenset({CHAT_COMPLETIONS, RESPONSES, EMBEDDINGS, MODELS})


def require_provider_module(module_name: str) -> ModuleType:
    """Import a provider SDK, failing instead of skipping in the unit-test environment.

    Importing (rather than ``pytest.importorskip``) keeps collection working in the pixi
    environments used by ``pytest -m env`` runs, while guaranteeing that the required
    provider coverage cannot silently disappear from the CPU unit-test job.

    Only an absent SDK is skippable. An import that fails *inside* an installed SDK
    propagates, so a broken install cannot masquerade as a missing one.
    """
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Python reports the outermost missing package, so ``google.genai.types`` surfaces
        # as ``google``; anything outside the requested path is a real failure, not absence.
        missing_name = exc.name or ""
        if missing_name != module_name and not module_name.startswith(f"{missing_name}."):
            raise
        current_env = pixi_utils.get_env_name()
        if current_env == UNIT_TEST_PIXI_ENV_NAME:
            pytest.fail(
                f"{module_name!r} must be installed in the {UNIT_TEST_PIXI_ENV_NAME!r} pixi environment "
                f"so provider coverage runs in the CPU unit-test job, but importing it failed: {exc}"
            )
        pytest.skip(f"{module_name!r} is not installed in the {current_env or 'active'} environment")
    else:
        return module


@dataclass
class StubRequest:
    """A single request captured by :class:`OpenAIStubServer`."""

    method: str
    path: str
    body: dict[str, Any] | None


def _default_chat_completion(content: str = "stub caption", *, model: str = "stub-model") -> dict[str, Any]:
    """Build a minimal, SDK-parseable ``chat.completion`` payload."""
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}},
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _default_responses(text: str = "stub response", *, model: str = "stub-model") -> dict[str, Any]:
    """Build a minimal, SDK-parseable Responses-API payload (``output_text``)."""
    return {
        "id": "resp-stub",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": model,
        "parallel_tool_calls": False,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "output": [
            {
                "type": "message",
                "id": "msg-stub",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        ],
    }


def _default_embedding(embedding: list[float] | None = None, *, model: str = "stub-model") -> dict[str, Any]:
    """Build a minimal, SDK-parseable ``CreateEmbeddingResponse`` payload."""
    vector = embedding if embedding is not None else [0.1, 0.2, 0.3]
    return {
        "object": "list",
        "model": model,
        "data": [{"object": "embedding", "index": 0, "embedding": vector}],
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    }


def _default_models(model_ids: list[str] | None = None) -> dict[str, Any]:
    """Build a minimal, SDK-parseable ``models.list`` page."""
    ids = model_ids if model_ids is not None else ["stub-model"]
    return {
        "object": "list",
        "data": [{"id": mid, "object": "model", "created": 0, "owned_by": "stub"} for mid in ids],
    }


class OpenAIStubServer:
    """Threadsafe handle over a localhost OpenAI-compatible HTTP stub server.

    Holds the canned response payloads (overridable per endpoint) and the list
    of received requests. Drive the real ``openai`` SDK against it by setting an
    endpoint's ``base_url`` to :attr:`base_url`.
    """

    def __init__(self, host: str, port: int) -> None:
        """Initialize with the bound host/port and the default canned responses."""
        self._host = host
        self._port = port
        self._lock = threading.Lock()
        self.requests: list[StubRequest] = []
        self._responses: dict[str, dict[str, Any]] = {
            CHAT_COMPLETIONS: _default_chat_completion(),
            RESPONSES: _default_responses(),
            EMBEDDINGS: _default_embedding(),
            MODELS: _default_models(),
        }

    @property
    def base_url(self) -> str:
        """Return the OpenAI ``base_url`` (includes the ``/v1`` suffix)."""
        return f"http://{self._host}:{self._port}/v1"

    def set_response(self, endpoint: str, payload: dict[str, Any]) -> None:
        """Override the raw canned payload for one endpoint (see module constants)."""
        if endpoint not in _VALID_ENDPOINTS:
            msg = f"Unknown stub endpoint {endpoint!r}; expected one of {sorted(_VALID_ENDPOINTS)}"
            raise ValueError(msg)
        with self._lock:
            self._responses[endpoint] = payload

    def set_chat_content(self, content: str, *, model: str = "stub-model") -> None:
        """Set the caption text returned by ``POST /v1/chat/completions``."""
        self.set_response(CHAT_COMPLETIONS, _default_chat_completion(content, model=model))

    def set_responses_text(self, text: str, *, model: str = "stub-model") -> None:
        """Set the ``output_text`` returned by ``POST /v1/responses``."""
        self.set_response(RESPONSES, _default_responses(text, model=model))

    def set_embedding(self, embedding: list[float], *, model: str = "stub-model") -> None:
        """Set the float vector returned by ``POST /v1/embeddings``."""
        self.set_response(EMBEDDINGS, _default_embedding(embedding, model=model))

    def set_models(self, model_ids: list[str]) -> None:
        """Set the model ids returned by ``GET /v1/models`` (drives ``model='auto'``)."""
        self.set_response(MODELS, _default_models(model_ids))

    def requests_for(self, path_suffix: str) -> list[StubRequest]:
        """Return captured requests whose path ends with ``path_suffix``."""
        with self._lock:
            return [req for req in self.requests if req.path.split("?", 1)[0].endswith(path_suffix)]

    def _record(self, request: StubRequest) -> None:
        with self._lock:
            self.requests.append(request)

    def _response_for(self, endpoint: str) -> dict[str, Any]:
        with self._lock:
            return self._responses[endpoint]


def _make_handler(stub: OpenAIStubServer) -> type[BaseHTTPRequestHandler]:  # noqa: C901  (routing branches)
    """Build a request handler bound to one :class:`OpenAIStubServer`."""

    class _Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any] | None:
            length = int(self.headers.get("Content-Length", "0"))
            if length == 0:
                return None
            raw = self.rfile.read(length)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return None
            return parsed if isinstance(parsed, dict) else None

        def _endpoint_for_path(self) -> str | None:
            path = self.path.split("?", 1)[0]
            if path.endswith("/chat/completions"):
                return CHAT_COMPLETIONS
            if path.endswith("/responses"):
                return RESPONSES
            if path.endswith("/embeddings"):
                return EMBEDDINGS
            if path.endswith("/models"):
                return MODELS
            return None

        def do_POST(self) -> None:
            body = self._read_json_body()
            stub._record(StubRequest(method="POST", path=self.path, body=body))
            endpoint = self._endpoint_for_path()
            if endpoint is None or endpoint == MODELS:
                self._send_json({"error": {"message": f"unhandled path {self.path}"}}, status=404)
                return
            self._send_json(stub._response_for(endpoint))

        def do_GET(self) -> None:
            stub._record(StubRequest(method="GET", path=self.path, body=None))
            if self._endpoint_for_path() == MODELS:
                self._send_json(stub._response_for(MODELS))
                return
            self._send_json({"error": {"message": f"unhandled path {self.path}"}}, status=404)

        def log_message(self, _format: str, *_args: object) -> None:
            """Silence per-request stderr logging."""

    return _Handler


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


@contextlib.contextmanager
def openai_stub_server() -> Iterator[OpenAIStubServer]:
    """Start an :class:`OpenAIStubServer` on an ephemeral localhost port.

    Yields the handle for the duration of the context and shuts the server down
    cleanly on exit. Bind to port 0 so concurrent tests never collide.
    """
    host = "127.0.0.1"
    server = _ThreadedHTTPServer((host, 0), BaseHTTPRequestHandler)
    stub = OpenAIStubServer(host, server.server_address[1])
    server.RequestHandlerClass = _make_handler(stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@dataclass
class GeminiStub:
    """A fake ``genai.Client`` plus the kwargs captured from generate calls.

    ``client`` mimics the real ``genai.Client``: ``client.models.generate_content``
    is sync and ``client.aio.models.generate_content`` is async, both returning
    the same configurable ``SimpleNamespace`` response. ``calls`` records the
    kwargs passed to either path so tests can assert on ``model`` / ``contents`` /
    ``config``.
    """

    client: SimpleNamespace
    response: SimpleNamespace
    calls: list[dict[str, Any]] = field(default_factory=list)
    async_generate: AsyncMock | None = None
    sync_generate: MagicMock | None = None


def make_gemini_stub(
    *,
    text: str | None = "stub gemini caption",
    candidates: list[Any] | None = None,
    usage_metadata: Any | None = None,  # noqa: ANN401  (mirrors SDK's untyped usage object)
) -> GeminiStub:
    """Build a :class:`GeminiStub` returning a configurable Gemini-style response.

    The response exposes the attributes the normalizers read: ``text``,
    ``candidates`` and ``usage_metadata``. Inject it by patching the stage's
    ``genai`` symbol so ``genai.Client(...)`` returns ``stub.client`` (this also
    exercises ``stage_setup``), or attach ``stub.client.aio`` directly.
    """
    response = SimpleNamespace(text=text, candidates=candidates or [], usage_metadata=usage_metadata)
    calls: list[dict[str, Any]] = []

    def _sync_generate(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return response

    async def _async_generate(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return response

    sync_mock = MagicMock(side_effect=_sync_generate)
    async_mock = AsyncMock(side_effect=_async_generate)
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=sync_mock),
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=async_mock)),
    )
    return GeminiStub(
        client=client,
        response=response,
        calls=calls,
        async_generate=async_mock,
        sync_generate=sync_mock,
    )
