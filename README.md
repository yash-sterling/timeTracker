# Time Tracker

Automatic screen activity tracker that runs locally on macOS. Takes a screenshot every 2 minutes, uses a local vision LLM (via Ollama) to describe what you're working on, and creates 15-minute summary events on your Google Calendar — all without sending data to the cloud.

## How it works

Two pipelines run in a single loop:

1. **Every 2 minutes** — takes a screenshot, sends it to a local vision model, and stores a short summary in a rolling buffer (`summaries.json`, last 30 minutes).
2. **Every 15 minutes** (aligned to :00, :15, :30, :45) — aggregates the individual summaries from the completed block, produces an overall subject + description, and creates a Google Calendar event.

All LLM prompts explicitly instruct the model to strip PII (names, emails, phone numbers, etc.) and replace them with placeholders.

## Prerequisites

- **macOS** (uses native `screencapture`)
- **Python 3.10+**
- **Ollama** with a vision-capable model
- **Google Cloud project** with Calendar API enabled (for calendar integration)

## Setup

### 1. Install Ollama

Download and install Ollama from [ollama.com/download](https://ollama.com/download). Open the app — it runs the server automatically.

Then pull and verify a vision-capable model (the default config uses `qwen3.5:9b`):

```bash
ollama run qwen3.5:9b
```

### 2. Clone and install dependencies

```bash
git clone <repo-url> timeTracker
cd timeTracker

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### 3. Set up Google Calendar API

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project (or select an existing one).
3. Navigate to **APIs & Services → Library**, search for **Google Calendar API**, and click **Enable**.
4. Go to **APIs & Services → Credentials**.
5. Click **Create Credentials → OAuth 2.0 Client ID**.
6. If prompted, configure the OAuth consent screen first:
   - Choose **External** user type.
   - Fill in the required fields (app name, support email).
   - Add the scope `https://www.googleapis.com/auth/calendar.events`.
   - Add your Google account as a test user.
7. Back in Credentials, create an **OAuth 2.0 Client ID**:
   - Application type: **Desktop app**
   - Name: anything you like
8. Download the JSON file and save it as `credentials.json` in the project directory.

9. **Publish the app** so your refresh token doesn't expire after 7 days:
   - Go to **OAuth consent screen → Audience** and click **Publish App**.
   - This is safe for a personal tool — it just prevents Google from revoking your token weekly.

On first run, the script will open your browser to complete the OAuth consent flow. After authorizing, a `token.json` file is saved so you won't need to authorize again. If your token does expire, just delete `token.json` and run the script again to re-authorize.

### 4. Grant screen recording permission

macOS requires permission for screen capture. The first time `screencapture` runs from your terminal, you may be prompted to allow it in **System Settings → Privacy & Security → Screen Recording**. Make sure your terminal app (Terminal, iTerm2, etc.) is listed and enabled.

## Quick start

```bash
source env/bin/activate

# Debug mode — single screenshot + summary, no loop, no calendar
python tracker.py --once

# Full mode — 2-min captures + 15-min calendar events
python tracker.py
```

Press `Ctrl+C` to stop the tracker.

## Configuration

All settings can be overridden with environment variables:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Model name (must support vision) |
| `TRACKER_CAPTURE_INTERVAL` | `120` | Seconds between screenshots |

Example:

```bash
OLLAMA_MODEL=qwen3.5:4b TRACKER_CAPTURE_INTERVAL=60 python tracker.py
```

## Files

| File | Purpose |
|---|---|
| `tracker.py` | Main script |
| `requirements.txt` | Python dependencies |
| `credentials.json` | Google OAuth client secret (you create this) |
| `token.json` | Cached OAuth token (auto-generated on first auth) |
| `summaries.json` | Rolling buffer of individual screenshot summaries (last 30 min) |
| `activity_log.jsonl` | Append-only log of calendar events created |

## Troubleshooting

**Ollama not responding** — make sure `ollama serve` is running and the model is pulled (`ollama list`).

**Empty LLM response** — some thinking models (like Qwen 3.5) put output in a `thinking` field instead of `response` when `format: json` is used. The script handles this automatically.

**Calendar auth fails** — delete `token.json` and run again to re-authorize. Make sure `credentials.json` is a valid OAuth 2.0 Desktop app credential.

**Screen capture is blank** — grant screen recording permission to your terminal in System Settings → Privacy & Security → Screen Recording.
