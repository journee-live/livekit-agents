# LiveKit Plugins Ojin

Agent Framework plugin for [Ojin](https://ojin.ai) avatars.

The plugin drives an Ojin Speech-To-Video model with the agent's TTS audio and
publishes the resulting lip-synced avatar — video plus the agent's own audio — on
the agent's participant. Nothing else joins the room.

## Installation

```bash
pip install livekit-plugins-ojin
```

## Pre-requisites

An Ojin API key and the config id of the persona to drive, both from your
[Ojin account](https://ojin.ai). Set them as `OJIN_API_KEY` and `OJIN_CONFIG_ID`,
or pass them to the constructor.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import ojin

session = AgentSession(...)

avatar = ojin.AvatarSession(
    # api_key / config_id default to OJIN_API_KEY / OJIN_CONFIG_ID
)
# Start the avatar first: it provides the session's audio output.
await avatar.start(session, room=ctx.room)

await session.start(agent=..., room=ctx.room)
```

See the [example agent](https://github.com/livekit/agents/blob/main/examples/avatar_agents/ojin/agent_worker.py).

## Notes

**Frame size comes from the model.** Ojin models publish at their own resolution
(1024x1024 and 736x1216 both occur), so the plugin waits for the first frame
before it publishes the video track and logs the size it found. A model that
changed resolution mid-session would corrupt the track, so mismatched frames are
dropped with an error rather than published.

**Interruptions stop the room audio immediately.** Ojin fades the cancelled turn
out server-side, which keeps the avatar's mouth closing naturally, but that fade
is not played into the room — `clear_buffer` means stop now. One case cannot be
cancelled yet: a barge-in that lands before the avatar has started speaking
(while the model is still rendering) has nothing to cancel, so that reply plays
through. It is logged once per session and is fixed by an upcoming SDK change.

**Tuning.** Pass an `ojin.stv.STVConfig` as `stv_config` to adjust buffering, the
interrupt fade, or the frame rate; the plugin keeps the video track's frame rate
in step with it.

**macOS.** `ojin-client[stv]` brings `opencv-python-headless`, whose bundled
ffmpeg libraries collide with the ones in `av`. The resulting `objc[...] Class
AVF... is implemented in both ...` warnings at startup are harmless.

## Configuration

| Argument | Default | Meaning |
|---|---|---|
| `api_key` | `OJIN_API_KEY` | Ojin API key |
| `config_id` | `OJIN_CONFIG_ID` | persona to drive |
| `ws_url` | `OJIN_WS_URL` or the SDK default | Ojin realtime endpoint |
| `stv_config` | SDK defaults | `STVConfig` passed to the client |
| `audio_sample_rate` | `24000` | rate the agent's TTS is resampled to |
| `session_ready_timeout` | `30.0` | seconds to wait for the session |
| `first_frame_timeout` | `15.0` | seconds to wait for the first video frame (waited *after* the one above, so an unreachable backend fails `start()` in up to 45 s) |
| `watchdog_timeout` | `10.0` | seconds without a server frame before failing |
| `turn_render_timeout` | `10.0` | seconds a fed turn may go unrendered |

If the session fails mid-conversation the avatar is torn down and the agent keeps
running (silently — the avatar carried the audio track) rather than hanging.
