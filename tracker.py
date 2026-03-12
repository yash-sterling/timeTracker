#!/usr/bin/env python3
"""
Periodic screen activity tracker.

Two pipelines:
  1. Every 2 minutes: screenshot → vision LLM → individual summary (stored in rolling buffer)
  2. Every 15 minutes (aligned to :00/:15/:30/:45): aggregate recent summaries →
     text LLM → overall subject + description → Google Calendar event
"""

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:9b")
CAPTURE_INTERVAL = int(os.environ.get("TRACKER_CAPTURE_INTERVAL", "120"))
BLOCK_MINUTES = 15
MAX_SUMMARY_AGE_MINUTES = 30

BASE_DIR = Path(__file__).parent
LOG_FILE = BASE_DIR / "activity_log.jsonl"
SUMMARIES_FILE = BASE_DIR / "summaries.json"
TOKEN_PATH = BASE_DIR / "token.json"
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
CONTEXT_PATH = BASE_DIR / "CONTEXT.txt"
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("tracker")
log.setLevel(logging.DEBUG)

SCREENSHOT_PROMPT = """You are an activity tracker. Look at this screenshot and describe in detail what the user is currently working on.

Be as specific as possible:
- Include application names, website names, document titles, file names, conversation participants, project names, and any other identifying details visible on screen.
- Describe both the high-level activity (e.g. "messaging a colleague") AND the specific content (e.g. "discussing the deployment timeline for the auth service with [person]").
- Include relevant details like URLs, channel names, repo names, branch names, error messages, etc.
- Do NOT redact or anonymize anything — full detail is needed for accurate summarization later.

Respond ONLY with valid JSON in this exact format (no extra text):
{"subject": "Brief 3-8 word topic", "description": "2-5 sentence detailed description of the activity"}"""

AGGREGATE_PROMPT_TEMPLATE = """You are an activity tracker. Below are individual activity summaries captured over the last 15 minutes:

{summaries}

Based on ALL of these summaries, do two things:

1. Produce a subject and description summarizing what the user did. Base this ONLY on the activity summaries above — do not incorporate any outside context.

2. Classify the focus level using the context below (if provided) to understand what counts as productive work for this user.
{context_section}
FOCUS CLASSIFICATION:
- "focused" — the user was primarily doing productive, intentional work (coding, writing, research, meetings, etc.)
- "low_focus" — a noticeable portion of the time was spent on distracting or non-essential activities such as: social media browsing (Twitter/X, Reddit, Instagram, TikTok, etc.), online shopping (Amazon, eBay, etc.), food delivery apps (DoorDash, UberEats, etc.), casual YouTube/video browsing, news feeds, or similar non-work activities

If even one or two summaries out of the block show distracting activity, classify as "low_focus".

IMPORTANT RULES:
- The subject and description must be based PURELY on the activity summaries. Do NOT reference or incorporate the user context into the subject or description.
- Do NOT include any Personally Identifiable Information (PII) such as real names, email addresses, phone numbers, physical addresses, account numbers, API keys, tokens, or passwords.
- Replace any PII with generic placeholders like [NAME], [EMAIL], [PHONE], etc.
- Synthesize the activities into a coherent summary — don't just list them.
- Be concise and factual.

Respond ONLY with valid JSON in this exact format (no extra text):
{{"subject": "Brief 3-8 word topic", "description": "1-3 sentence description of overall activity. If focus is low_focus, append a brief explanation of why (e.g. which distracting activities were detected).", "focus": "focused or low_focus"}}"""


# ---------------------------------------------------------------------------
# Input idle detection
# ---------------------------------------------------------------------------

def seconds_since_last_input() -> float:
    """Seconds since the last keyboard or mouse event (system-wide)."""
    import Quartz
    return Quartz.CGEventSourceSecondsSinceLastEventType(
        Quartz.kCGEventSourceStateCombinedSessionState,
        Quartz.kCGAnyInputEventType,
    )


# ---------------------------------------------------------------------------
# Active window detection
# ---------------------------------------------------------------------------

def get_active_window_id() -> int | None:
    """Get the CGWindowID of the frontmost application's window."""
    import Quartz

    try:
        result = subprocess.run(
            [
                "osascript", "-e",
                'tell application "System Events" to get name of first '
                'application process whose frontmost is true',
            ],
            capture_output=True, text=True, timeout=5,
        )
        frontmost_app = result.stdout.strip()
        if not frontmost_app:
            return None
        log.debug("Frontmost app: %s", frontmost_app)
    except Exception:
        log.debug("Could not determine frontmost app")
        return None

    windows = Quartz.CGWindowListCopyWindowInfo(
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListExcludeDesktopElements,
        Quartz.kCGNullWindowID,
    )
    for w in windows:
        if (
            w.get("kCGWindowOwnerName") == frontmost_app
            and w.get("kCGWindowLayer", 999) == 0
            and w.get("kCGWindowBounds", {}).get("Width", 0) > 50
        ):
            wid = int(w["kCGWindowNumber"])
            log.debug("Active window id=%d (%s)", wid, w.get("kCGWindowName", ""))
            return wid
    return None


# ---------------------------------------------------------------------------
# Screenshot & encoding
# ---------------------------------------------------------------------------

def take_screenshot() -> str:
    path = os.path.join(tempfile.gettempdir(), "timetracker_screenshot.png")
    window_id = get_active_window_id()
    if window_id:
        log.debug("Capturing active window (id=%d)", window_id)
        subprocess.run(["screencapture", "-x", f"-l{window_id}", path], check=True)
    else:
        log.debug("No active window found, falling back to full screen")
        subprocess.run(["screencapture", "-x", path], check=True)
    size_kb = os.path.getsize(path) / 1024
    log.info("Screenshot saved (%.0f KB)", size_kb)
    return path


def encode_image(path: str) -> str:
    log.debug("Encoding image to base64...")
    with open(path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode("utf-8")
    log.info("Image encoded (%.0f KB payload)", len(encoded) / 1024)
    return encoded


# ---------------------------------------------------------------------------
# Ollama helpers
# ---------------------------------------------------------------------------

def _parse_ollama_response(body: dict) -> str:
    """Extract text content from an Ollama response, handling thinking models."""
    raw = body.get("response", "")
    if not raw.strip() and body.get("thinking"):
        log.debug("'response' empty, using 'thinking' field instead")
        raw = body["thinking"]

    if body.get("eval_count"):
        log.debug(
            "Tokens — prompt: %s, eval: %s, eval_duration: %sms",
            body.get("prompt_eval_count", "?"),
            body.get("eval_count", "?"),
            round(body.get("eval_duration", 0) / 1e6),
        )

    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if cleaned != raw.strip():
        log.debug("Stripped <think> tags, cleaned length: %d chars", len(cleaned))
    return cleaned


def _parse_json_result(text: str) -> dict:
    """Parse a JSON subject+description from LLM text output."""
    try:
        parsed = json.loads(text)
        result = {
            "subject": str(parsed.get("subject", "Unknown"))[:120],
            "description": str(parsed.get("description", "Unknown"))[:500],
        }
        log.debug("Parsed JSON: %s", result)
        return result
    except json.JSONDecodeError as e:
        log.warning("Failed to parse JSON: %s — raw: %s", e, text[:200])
        return {"subject": "Parse error", "description": text[:200]}


def describe_screenshot(image_b64: str) -> dict:
    """Vision LLM call: screenshot → individual subject + description."""
    log.info("Sending screenshot to Ollama (model=%s)...", MODEL)
    start = time.monotonic()

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": SCREENSHOT_PROMPT,
            "images": [image_b64],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "num_predict": 256},
        },
        timeout=120,
    )
    elapsed = time.monotonic() - start
    log.info("Screenshot LLM responded in %.1fs (HTTP %d)", elapsed, resp.status_code)
    resp.raise_for_status()

    text = _parse_ollama_response(resp.json())
    log.debug("Screenshot raw (%d chars): %s", len(text), text[:300])
    return _parse_json_result(text)


def _load_user_context() -> str:
    """Load optional CONTEXT.txt for the aggregation prompt."""
    if CONTEXT_PATH.exists():
        try:
            text = CONTEXT_PATH.read_text().strip()
            if text:
                log.debug("Loaded user context (%d chars)", len(text))
                return f"\nContext about the user (use this to better interpret the activity):\n{text}\n"
        except OSError:
            pass
    return ""


def aggregate_summaries(summaries: list[dict]) -> dict:
    """Text LLM call: list of summaries → overall subject + description."""
    formatted = "\n".join(
        f"- [{s['timestamp']}] {s['subject']}: {s['description']}"
        for s in summaries
    )
    context_section = _load_user_context()
    prompt = AGGREGATE_PROMPT_TEMPLATE.format(summaries=formatted, context_section=context_section)

    log.info("Sending %d summaries to Ollama for aggregation...", len(summaries))
    start = time.monotonic()

    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "num_predict": 256},
        },
        timeout=120,
    )
    elapsed = time.monotonic() - start
    log.info("Aggregation LLM responded in %.1fs (HTTP %d)", elapsed, resp.status_code)
    resp.raise_for_status()

    text = _parse_ollama_response(resp.json())
    log.debug("Aggregation raw (%d chars): %s", len(text), text[:300])
    result = _parse_json_result(text)
    try:
        parsed = json.loads(text)
        result["focus"] = parsed.get("focus", "focused")
    except json.JSONDecodeError:
        result["focus"] = "focused"
    log.info("Focus level: %s", result["focus"])
    return result


# ---------------------------------------------------------------------------
# Rolling summary buffer (summaries.json)
# ---------------------------------------------------------------------------

def load_summaries() -> list[dict]:
    if SUMMARIES_FILE.exists():
        try:
            return json.loads(SUMMARIES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Could not read %s, starting fresh", SUMMARIES_FILE)
    return []


def save_summaries(summaries: list[dict]):
    SUMMARIES_FILE.write_text(json.dumps(summaries))


def add_summary(entry: dict):
    summaries = load_summaries()
    summaries.append(entry)
    cutoff = (datetime.now() - timedelta(minutes=MAX_SUMMARY_AGE_MINUTES)).isoformat()
    summaries = [s for s in summaries if s["timestamp"] >= cutoff]
    save_summaries(summaries)
    log.debug("Buffer: %d summaries (pruned to last %d min)", len(summaries), MAX_SUMMARY_AGE_MINUTES)


def get_summaries_in_range(start: datetime, end: datetime) -> list[dict]:
    summaries = load_summaries()
    start_iso, end_iso = start.isoformat(), end.isoformat()
    matched = [s for s in summaries if start_iso <= s["timestamp"] < end_iso]
    log.debug("Found %d summaries in range %s → %s", len(matched), start.strftime("%H:%M"), end.strftime("%H:%M"))
    return matched


# ---------------------------------------------------------------------------
# Google Calendar
# ---------------------------------------------------------------------------

def get_local_timezone() -> str:
    try:
        link = os.readlink("/etc/localtime")
        if "/zoneinfo/" in link:
            return link.split("/zoneinfo/")[-1]
    except OSError:
        pass
    return "UTC"


def get_calendar_service():
    from google.auth.transport.requests import Request as GoogleAuthRequest
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None
    if TOKEN_PATH.exists():
        log.debug("Loading existing token from %s", TOKEN_PATH)
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            log.info("Token expired, refreshing...")
            creds.refresh(GoogleAuthRequest())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(
                    f"Missing {CREDENTIALS_PATH}. Download OAuth 2.0 Client ID "
                    "(Desktop app) from Google Cloud Console → APIs & Services → "
                    "Credentials, then save the JSON as credentials.json"
                )
            log.info("No valid token — launching browser for OAuth consent...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH), SCOPES
            )
            creds = flow.run_local_server(port=0)

        TOKEN_PATH.write_text(creds.to_json())
        log.info("Saved auth token to %s", TOKEN_PATH)

    service = build("calendar", "v3", credentials=creds)
    log.info("Google Calendar service ready")
    return service


def create_calendar_event(
    service, subject: str, description: str, start: datetime, end: datetime,
    color_id: str | None = None,
):
    tz = get_local_timezone()
    event_body = {
        "summary": subject,
        "description": description,
        "start": {"dateTime": start.isoformat(), "timeZone": tz},
        "end": {"dateTime": end.isoformat(), "timeZone": tz},
    }
    if color_id:
        event_body["colorId"] = color_id
    log.info(
        "Creating calendar event: '%s' (%s → %s, tz=%s)",
        subject, start.strftime("%H:%M"), end.strftime("%H:%M"), tz,
    )
    created = (
        service.events().insert(calendarId="primary", body=event_body).execute()
    )
    log.info("Calendar event created: %s", created.get("htmlLink"))
    return created


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def block_key(dt: datetime) -> str:
    """Unique key for a 15-minute block, e.g. '2026-03-03 05:15'."""
    minute = (dt.minute // BLOCK_MINUTES) * BLOCK_MINUTES
    return dt.replace(minute=minute, second=0, microsecond=0).strftime("%Y-%m-%d %H:%M")


def completed_block_times() -> tuple[datetime, datetime]:
    """Return (start, end) of the 15-minute block that most recently ended."""
    now = datetime.now()
    quarter = (now.minute // BLOCK_MINUTES) * BLOCK_MINUTES
    block_end = now.replace(minute=quarter, second=0, microsecond=0)
    block_start = block_end - timedelta(minutes=BLOCK_MINUTES)
    return block_start, block_end


# ---------------------------------------------------------------------------
# Activity log (calendar events)
# ---------------------------------------------------------------------------

def log_entry(entry: dict):
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.debug("Entry written to %s", LOG_FILE)


# ---------------------------------------------------------------------------
# Core capture (single screenshot → summary)
# ---------------------------------------------------------------------------

def capture_screenshot() -> dict:
    """Take a screenshot, describe it with the vision LLM, return the summary."""
    timestamp = datetime.now().isoformat()
    log.info("--- Screenshot capture ---")

    screenshot_path = take_screenshot()
    image_b64 = encode_image(screenshot_path)
    result = describe_screenshot(image_b64)

    entry = {
        "timestamp": timestamp,
        "subject": result["subject"],
        "description": result["description"],
    }

    log.info("  Subject:     %s", entry["subject"])
    log.info("  Description: %s", entry["description"])

    os.remove(screenshot_path)
    log.debug("Cleaned up screenshot file")
    return entry


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(once: bool = False):
    log.info("Time Tracker started")
    log.info("  Model: %s", MODEL)
    log.info("  Capture interval: %ds", CAPTURE_INTERVAL)
    log.info("  Calendar block: %dm", BLOCK_MINUTES)
    log.info("  Summary buffer: last %dm", MAX_SUMMARY_AGE_MINUTES)
    log.info("  Log file: %s", LOG_FILE)

    if once:
        entry = capture_screenshot()
        add_summary(entry)
        return

    service = get_calendar_service()
    last_calendar_block = block_key(datetime.now())
    log.info("Starting in block %s — entering main loop", last_calendar_block)

    while True:
        try:
            # --- Check for 15-minute boundary crossing ---
            current_block = block_key(datetime.now())
            if current_block != last_calendar_block:
                block_start, block_end = completed_block_times()
                recent = get_summaries_in_range(block_start, block_end)

                if recent:
                    all_inactive = all(s["subject"] == "No activity" for s in recent)

                    if all_inactive:
                        log.info(
                            "=== 15-min boundary (%s → %s) — all %d summaries inactive, skipping LLM ===",
                            block_start.strftime("%H:%M"), block_end.strftime("%H:%M"), len(recent),
                        )
                        result = {"subject": "No activity", "description": "No keyboard or mouse activity detected during this period."}
                        color_id = "8"  # graphite
                    else:
                        log.info(
                            "=== 15-min boundary (%s → %s) — aggregating %d summaries ===",
                            block_start.strftime("%H:%M"), block_end.strftime("%H:%M"), len(recent),
                        )
                        result = aggregate_summaries(recent)
                        color_id = "5" if result.get("focus") == "low_focus" else None  # banana for low focus

                    create_calendar_event(
                        service, result["subject"], result["description"],
                        block_start, block_end, color_id=color_id,
                    )
                    log_entry({
                        "timestamp": datetime.now().isoformat(),
                        "block_start": block_start.isoformat(),
                        "block_end": block_end.isoformat(),
                        "subject": result["subject"],
                        "description": result["description"],
                        "source_summaries": len(recent),
                    })
                else:
                    log.info("15-min boundary crossed but no summaries in block")

                last_calendar_block = current_block

            # --- Screenshot + individual summary (every ~2 min) ---
            idle = seconds_since_last_input()
            log.debug("Idle time: %.0fs (threshold: %ds)", idle, CAPTURE_INTERVAL)

            if idle >= CAPTURE_INTERVAL:
                log.info("No keyboard/mouse activity for %ds — skipping screenshot", int(idle))
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "subject": "No activity",
                    "description": "No keyboard or mouse activity detected.",
                }
            else:
                entry = capture_screenshot()
            add_summary(entry)

        except KeyboardInterrupt:
            log.info("Stopped by user.")
            break
        except Exception:
            log.exception("Error during cycle")

        log.info("Sleeping %ds until next capture...\n", CAPTURE_INTERVAL)
        time.sleep(CAPTURE_INTERVAL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Screen activity tracker")
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single screenshot capture and exit (debug mode)",
    )
    args = parser.parse_args()
    run(once=args.once)
