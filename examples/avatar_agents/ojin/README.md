# Ojin avatar example

Runs a voice agent whose speech drives an [Ojin](https://ojin.ai) avatar. The
plugin renders the avatar in process and publishes it on the agent's own
participant, so no second participant joins the room.

## Setup

```bash
pip install "livekit-agents[openai]~=1.5" livekit-plugins-ojin python-dotenv
```

`.env`:

```
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
OPENAI_API_KEY=...

OJIN_API_KEY=...
OJIN_CONFIG_ID=...     # the persona to drive
# OJIN_WS_URL=...      # optional, defaults to wss://models.ojin.ai/realtime
```

The config id must name a model that has a live backend. If it does not, the
session fails to start with `BACKEND_UNAVAILABLE` — that is the usual symptom of
a config pointing at a renamed or undeployed model, not of a bad key.

## Run

```bash
python agent_worker.py dev
```

## What to check

- The avatar animates and lip-syncs while the agent speaks.
- Three consecutive turns complete: the session returns to listening each time.
- Interrupting mid-utterance stops the room audio immediately (the avatar's video
  fades out) and the next reply plays normally.
- Interrupting at the very start of a reply — before the avatar has begun
  speaking — cannot be cancelled yet, so that reply plays through. This is a
  known Ojin SDK limitation and is logged once per session.

The avatar's frame size comes from the model, so it is logged at startup rather
than configured here (1024x1024 and 736x1216 both occur in practice).
