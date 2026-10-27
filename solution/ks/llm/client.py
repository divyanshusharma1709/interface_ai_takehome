"""The LLM transport and session: settings, budget, response cache, API client, request lifecycle."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from ..core.tokens import tokenize

# ---- env
SOLUTION_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "claude-haiku-5-5"
MODES = ("off", "live", "record", "replay")
LIVE_MODES = ("live", "record")  # the modes that may call the API


def load_dotenv(*folders) -> None:
    """Read KEY=VALUE lines from a `.env` in each folder (never overriding an existing variable."""
    for folder in folders:
        try:
            p = Path(folder) / ".env"
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip().removeprefix("export ").strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except OSError:
            continue


def sdk_available() -> bool:
    try:
        return importlib.util.find_spec("anthropic") is not None
    except (ImportError, ValueError):  # a module object without a usable __spec__
        return False


@dataclass(frozen=True)
class LLMSettings:
    """What a parser / reader runs with."""

    mode: str
    model: str
    cache_path: Path
    client: object = None

    @property
    def active(self) -> bool:
        return self.mode != "off"

    @classmethod
    def off(cls) -> "LLMSettings":
        return cls("off", DEFAULT_MODEL, SOLUTION_DIR / ".llm_cache.jsonl")

    @classmethod
    def resolve(
        cls, mode: str | None = None, model: str | None = None, cache_path=None, client=None, folder=None
    ) -> "LLMSettings":
        load_dotenv(*(p for p in (folder and Path(folder).parent, folder, Path.cwd()) if p))
        can_call = client is not None or (bool(os.environ.get("ANTHROPIC_API_KEY")) and sdk_available())
        mode = (mode or os.environ.get("KS_LLM") or "").strip().lower()
        if mode not in MODES:
            mode = "live" if can_call else "off"
        if mode in LIVE_MODES and not can_call:
            mode = "off"
        default_cache = (Path(folder) if folder else SOLUTION_DIR) / ".llm_cache.jsonl"
        return cls(
            mode,
            model or os.environ.get("KS_LLM_MODEL") or DEFAULT_MODEL,
            Path(cache_path or os.environ.get("KS_LLM_CACHE") or default_cache),
            client,
        )


# ---- budget
DEFAULT_TOTAL = 3.0


class Budget:
    """Started when answer() starts (rule-path time counts), shared by parser and reader."""

    MIN_LEFT = 0.5

    def __init__(self, total: float | None = None):
        try:
            self.total = float(total if total is not None else os.environ.get("KS_LLM_BUDGET", DEFAULT_TOTAL))
        except ValueError:
            self.total = DEFAULT_TOTAL
        self._local = threading.local()  # per thread: answers may run concurrently

    def start(self) -> None:
        self._local.t0 = time.perf_counter()

    def remaining(self) -> float:
        t0 = getattr(self._local, "t0", None)
        return float("inf") if t0 is None else self.total - (time.perf_counter() - t0)

    def exhausted(self) -> bool:
        return self.remaining() < self.MIN_LEFT


# ---- cache
_QUESTION_WORDS = re.compile(r"\b(?:where|when|who|whom|whose|why|how|which|what|can|does|is|are|do|will)\b")


def normalize_question(q: str) -> str:
    """Lowercase, whitespace collapsed (the part of the disk cache key that depends on the question)."""
    return " ".join((q or "").lower().split())


def memo_key(q: str) -> str:
    """Lowercase, punctuation / stopwords / fillers dropped, words stemmed and sorted."""
    toks = sorted(set(tokenize(q or "")) | {"?" + w for w in _QUESTION_WORDS.findall((q or "").lower())})
    return " ".join(toks) if toks else normalize_question(q)


class ResponseCache:
    """Reads the cache file lazily (once); appends a line per recorded response."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._responses: dict[str, dict] | None = None
        self._fixtures: dict[tuple, dict] = {}
        self._lock = threading.Lock()

    def lookup(self, key: str, memo: tuple, also_fixture: tuple | None = None):
        """The recorded response for a disk key, else the fixture line for the memo key (or `also_fixture`)."""
        self._ensure_loaded()
        hit = self._responses.get(key)
        if hit is None:
            hit = self._fixtures.get(memo)
        if hit is None and also_fixture is not None:
            hit = self._fixtures.get(also_fixture)
        return hit

    def record(self, key: str, version: str, model: str, question: str, as_of, response) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "key": key,
                "v": version,
                "model": model,
                "q": question,
                "as_of": str(as_of or ""),
                "response": response,
            }
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._responses is None:
                self._responses = self._read()

    def _read(self) -> dict:
        responses: dict = {}
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return responses
        for line in lines:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("fixture"):
                self._fixtures[(memo_key(r.get("q", "")), str(r.get("as_of") or ""))] = r.get("response")
            elif r.get("key"):
                responses[r["key"]] = r.get("response")
        return responses


# ---- client
# ---------------------------------------------------------------------------------------------------- errors
class ToolChoiceRejected(Exception):
    """The API refuses forced tool use for this model: nothing useful can be asked of it."""


class TruncatedOutput(Exception):
    """The reply hit max_tokens: a cut-off tool call is never trusted."""


def api_message(e: BaseException) -> str:
    """The API's own error text (body.error.message) when the exception carries it, else str(e)."""
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(getattr(e, "message", None) or e)


_STATUS_CLASS = (
    (401, "auth"),
    (403, "auth"),
    (400, "bad_request"),
    (422, "bad_request"),
    (404, "not_found"),
    (429, "rate_limit"),
)
_NAME_CLASS = {
    "AuthenticationError": "auth",
    "PermissionDeniedError": "auth",
    "BadRequestError": "bad_request",
    "UnprocessableEntityError": "bad_request",
    "NotFoundError": "not_found",
    "RateLimitError": "rate_limit",
    "InternalServerError": "server",
    "OverloadedError": "server",
}


def classify_error(e: BaseException) -> str:
    """auth | bad_request | not_found | rate_limit | server | timeout | invalid_output | truncated | network |."""
    name = type(e).__name__
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    api_json = isinstance(body, dict) and ("error" in body or "type" in body)
    if "Timeout" in name or isinstance(e, TimeoutError):
        return "timeout"
    if isinstance(status, int):
        if not api_json and (status in (400, 401, 403, 404, 422, 429) or (status >= 500 and status != 529)):
            return "network"  # a proxy / gateway answered, not the API
        for code, cls in _STATUS_CLASS:
            if status == code:
                return cls
        if status >= 500:
            return "server"
    if name in _NAME_CLASS:
        return _NAME_CLASS[name]
    if isinstance(e, TruncatedOutput):
        return "truncated"
    if isinstance(e, (ValueError, KeyError, TypeError)) or "JSON" in name:
        return "invalid_output"
    if "Connection" in name or "SSL" in name or "Socket" in name or "Connect" in name or isinstance(e, OSError):
        return "network"
    return "unknown"


_NEVER_DROP = ("tools", "tool_choice", "messages", "model", "max_tokens")
_PARAM_PATTERNS = (
    r"`(\w+)`\s+(?:is|are)\s+(?:deprecated|not supported|unsupported|not allowed)",
    r"(?:^|[\s.])(\w+)\s*:\s*Extra inputs are not permitted",
    r"unsupported parameter[s]?[:\s]+[`'\"]?(\w+)",
    r"(?:unknown|unrecognized|unexpected) (?:parameter|field|argument)[:\s]+[`'\"]?(\w+)",
    r"[`'\"](\w+)[`'\"]\s+(?:is|are)\s+(?:not supported|unsupported|deprecated|not allowed)",
    r"does not support (?:the )?(?:parameter )?[`'\"]?(\w+)",
)


def rejected_param(message: str) -> str | None:
    """The request parameter a 400 message names as unsupported / deprecated, or None."""
    for rx in _PARAM_PATTERNS:
        m = re.search(rx, message or "", re.I)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------------------------------- transport
MAX_ATTEMPTS = 4
_USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")


class LLMClient:
    """One SDK client per parser / reader."""

    def __init__(self, model: str, timeout: float, sdk=None):
        self.model = model
        self.timeout = timeout
        self.sdk = sdk
        self.unsupported: set = set()
        self.last_usage: dict = {}
        self.usage_totals: dict = {}

    def call_bounded(self, system: str, tool: dict, user: str, max_tokens: int, timeout: float) -> dict:
        """`call` with a hard wall-clock limit."""
        box: dict = {}

        def run():
            try:
                box["out"] = self.call(system, tool, user, max_tokens, timeout)
            except BaseException as e:
                box["err"] = e

        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout + 0.05)
        if th.is_alive():
            raise TimeoutError("Request timed out (wall-clock limit)")
        if "err" in box:
            raise box["err"]
        return box.get("out")

    def call(self, system: str, tool: dict, user: str, max_tokens: int, timeout: float | None = None) -> dict:
        """The tool input the model returned (raises on API errors."""
        if self.sdk is None:
            import anthropic  # lazy: never imported when the LLM is off

            self.sdk = anthropic.Anthropic(max_retries=0, timeout=self.timeout)
        kw = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=self._system_blocks(system, cached=True),
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[{"role": "user", "content": user}],
            timeout=timeout or self.timeout,
        )
        kw = {k: v for k, v in kw.items() if k not in self.unsupported}
        if "cache_control" in self.unsupported:
            kw["system"] = self._system_blocks(system, cached=False)
        msg = self._create(kw, system)
        self._note_usage(msg)
        if getattr(msg, "stop_reason", None) == "max_tokens":
            raise TruncatedOutput(f"output hit max_tokens={max_tokens}")
        return self._tool_input(msg)

    @staticmethod
    def _system_blocks(text: str, cached: bool) -> list:
        block = {"type": "text", "text": text}
        return [dict(block, cache_control={"type": "ephemeral"})] if cached else [block]

    def _create(self, kw: dict, system: str):
        for _ in range(MAX_ATTEMPTS):
            try:
                return self.sdk.messages.create(**kw)
            except Exception as e:
                param = self._rejected(e)
                if param == "tool_choice":
                    raise ToolChoiceRejected(
                        f"the API rejected tool_choice for model {self.model}: {api_message(e)}"
                    ) from e
                if not param or param in _NEVER_DROP or (param not in kw and param != "cache_control"):
                    raise
                self.unsupported.add(param)  # later calls skip the failed round trip
                kw.pop(param, None)
                if param == "cache_control":
                    kw["system"] = self._system_blocks(system, cached=False)
        raise ValueError("the API kept rejecting request parameters")

    @staticmethod
    def _rejected(e: Exception) -> str | None:
        if isinstance(e, TypeError):
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            return m.group(1) if m else None
        if classify_error(e) == "bad_request":
            return rejected_param(api_message(e))
        raise e

    def _note_usage(self, msg) -> None:
        u = getattr(msg, "usage", None)
        self.last_usage = {k: int(getattr(u, k, 0) or 0) for k in _USAGE_FIELDS} if u else {}
        for k, val in self.last_usage.items():
            self.usage_totals[k] = self.usage_totals.get(k, 0) + val

    @staticmethod
    def _tool_input(msg) -> dict:
        for block in getattr(msg, "content", []) or []:
            if getattr(block, "type", None) == "tool_use":
                inp = getattr(block, "input", None)
                return json.loads(inp) if isinstance(inp, str) else inp
        raise ValueError("no tool_use block in the response")


# ---- session
STATE_LIMIT = 20_000
LATENCY_LIMIT = 2_000
FATAL_AFTER = 5
PERMANENT = ("auth", "not_found", "network", "bad_request")


class BoundedMap(OrderedDict):
    """A dict that forgets its least recently set key beyond `limit`."""

    def __init__(self, limit: int = STATE_LIMIT):
        super().__init__()
        self.limit = limit

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self.limit:
            self.popitem(last=False)


class Session:
    """One prompt's request lifecycle; subclasses give version, max_tokens, system, tool, cache_key, user, validate."""

    def __init__(self, settings: LLMSettings, timeout: float):
        self.mode = settings.mode
        self.client = LLMClient(settings.model, timeout, settings.client)
        self.cache = ResponseCache(settings.cache_path)
        self.budget = Budget()
        self.budget_skipped = False
        self.fatal: str | None = None
        self.last_error = ""
        self.stats = {
            "calls": 0,
            "errors": 0,
            "hits": 0,
            "misses": 0,
            "used": 0,
            "error_classes": {},
            "latencies_ms": [],
            "usage": self.client.usage_totals,
        }
        self.info = BoundedMap()  # (memo key, as_of) -> {status, ms, usage}
        self.outcome = BoundedMap()  # (memo key, as_of) -> the cascade's verdict
        self._memo = BoundedMap()
        self._lock = threading.Lock()
        self._live_calls = self._permanent_failures = 0

    @property
    def model(self) -> str:
        return self.client.model

    @property
    def timeout(self) -> float:
        return self.client.timeout

    @property
    def active(self) -> bool:
        return self.mode != "off"

    def set_outcome(self, question: str, as_of, status: str) -> None:
        """Cascade bookkeeping: 'skipped:confident_rules' | 'used:rescue' | 'used:uncertain_parse' |."""
        self.outcome[(memo_key(question), str(as_of or ""))] = status
        if status.startswith("used"):
            self.stats["used"] += 1

    def info_for(self, question: str, as_of=None) -> dict:
        """{'status': 'used' | 'replay_hit' | 'fallback:<reason>' | 'off', 'ms'."""
        if self.mode == "off":
            return {"status": "off", "ms": None}
        mk = (memo_key(question), str(as_of or ""))
        base = dict(self.info.get(mk, {"status": "fallback:not_asked", "ms": None}))
        if mk in self.outcome:
            base["status"] = self.outcome[mk]
        return base

    def get(self, text: str, as_of=None, payload=None, fixture_text: str | None = None):
        """The validated reply for `text` (the cache-key text), or None."""
        if self.mode == "off":
            return None
        mk = (memo_key(text), str(as_of or ""))
        with self._lock:
            if mk in self._memo:
                return self._memo[mk]
        if self.fatal and self.mode in LIVE_MODES:
            self.info[mk] = {"status": f"fallback:{self.fatal}", "ms": None}
            return None
        self.budget_skipped = False
        hit = None
        if self.mode != "record":
            fixture = (memo_key(fixture_text), mk[1]) if fixture_text is not None else None
            hit = self.cache.lookup(self.cache_key(text, as_of, self.client.model), mk, fixture)
        if hit is not None:
            self.stats["hits"] += 1
            return self._finish(mk, hit, "replay_hit", None)
        if self.mode not in LIVE_MODES:
            self.stats["misses"] += 1
            return self._finish(mk, None, "fallback:miss", None)
        if self.budget.exhausted():
            self.stats["budget_skips"] = self.stats.get("budget_skips", 0) + 1
            self.budget_skipped = True
            with self._lock:
                self.info[mk] = {"status": "fallback:budget", "ms": None, "usage": {}}
            return None
        raw, status, ms = self._ask(text, as_of, payload)
        return self._finish(mk, raw, status, ms)

    def _ask(self, text: str, as_of, payload) -> tuple:
        t0 = time.perf_counter()
        raw, status = None, "pending"
        self.stats["calls"] += 1
        try:
            raw = self.client.call_bounded(
                self.system,
                self.tool,
                self.user(text, payload),
                self.max_tokens,
                min(self.client.timeout, self.budget.remaining()),
            )
        except Exception as e:
            cls = classify_error(e)
            self.stats["errors"] += 1
            self._count_error(cls)
            self.last_error = f"{type(e).__name__}: {api_message(e)[:300]}"
            status = f"fallback:{cls}"
        ms = round((time.perf_counter() - t0) * 1000)
        latencies = self.stats["latencies_ms"]
        latencies.append(ms)
        del latencies[:-LATENCY_LIMIT]
        self._track_permanent_failures(status)
        if raw is not None and self.mode == "record":
            self.cache.record(
                self.cache_key(text, as_of, self.client.model), self.version, self.client.model, text, as_of, raw
            )
        return raw, status, ms

    def _track_permanent_failures(self, status: str) -> None:
        self._live_calls += 1
        if status.startswith("fallback") and status.split(":")[1] in PERMANENT:
            self._permanent_failures += 1
        if self._live_calls == FATAL_AFTER and self._permanent_failures == FATAL_AFTER:
            self.fatal = status.split(":")[1]

    def _count_error(self, cls: str) -> None:
        self.stats["error_classes"][cls] = self.stats["error_classes"].get(cls, 0) + 1

    def _finish(self, mk: tuple, raw, status: str, ms):
        out = self.validate(raw) if raw is not None else None
        if raw is not None and not out:
            status = "fallback:invalid_output"
            self._count_error("invalid_output")
        keep_usage = ms is not None and (not status.startswith("fallback") or status.endswith("truncated"))
        with self._lock:
            self._memo[mk] = out or None
            self.info[mk] = {"status": status, "ms": ms, "usage": dict(self.client.last_usage) if keep_usage else {}}
        return out or None
