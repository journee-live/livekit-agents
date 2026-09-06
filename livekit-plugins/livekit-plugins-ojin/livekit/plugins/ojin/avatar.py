from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator

from livekit import rtc
from livekit.agents import NOT_GIVEN, AgentSession, NotGivenOr, utils
from livekit.agents.voice.avatar import (
    AudioReceiver,
    AudioSegmentEnd,
    AvatarOptions,
    AvatarRunner,
    AvatarSession as BaseAvatarSession,
    QueueAudioOutput,
    VideoGenerator,
)
from ojin.stv import (
    FrameType,
    OjinSTVClient,
    STVAudioFrame,
    STVConfig,
    STVEvent,
    STVVideoFrame,
)

from .errors import OjinException
from .frames import downmix_to_mono, is_silence, to_audio_frame, to_video_frame
from .log import logger

SAMPLE_RATE = 24000
NUM_CHANNELS = 1
VIDEO_FPS = 25

# ~half a second of video at 25 fps: a staleness cap, not a buffer. Audio is
# never dropped.
_VIDEO_QUEUE_SIZE = 12
_DEFAULT_FADE_S = 0.75
_DEFAULT_TURN_RENDER_TIMEOUT = 10.0
_DEFAULT_SESSION_READY_TIMEOUT = 30.0
# Covers server render plus the SDK's warm-up hold once video starts arriving.
_DEFAULT_FIRST_FRAME_TIMEOUT = 15.0
# Production has been seen pausing frames for ~11s on an otherwise healthy
# socket. The watchdog is terminal - there is no reconnect - so it has to sit
# above a plausible hiccup rather than at it.
_DEFAULT_WATCHDOG_TIMEOUT = 15.0

_FrameOrEnd = rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd


class _FrameSink:
    """The client's ``STVOutput``: where Ojin's 25 fps clock hands off to ours.

    Ojin calls ``write_audio``/``write_video`` from its playback loop, so nothing
    here may await the downstream consumer (on a slow model a blocking write costs
    half of a 40 ms tick). Frames go into one ordered deque that the runner drains
    on its own schedule; audio and segment markers are never dropped, video is
    capped and drops oldest-first.

    Demux: the SDK emits one audio frame per tick forever, zero-filled when no turn
    is playing, so only non-zero audio is forwarded. ``AudioSegmentEnd`` — the
    marker the runner translates into the playback-finished report the agent
    session blocks on — is emitted only when both the client's stop edge has fired
    and the input segment has closed, because the stop edge alone also fires on a
    mid-turn TTS underrun. On barge-in the sink purges audio and markers, keeps
    video, and mutes until the first START_OF_SPEECH video frame or a fade-length
    timer — never the stop edge, which fires one tick after ``interrupt()`` while
    the fade is still audible.
    """

    def __init__(
        self,
        *,
        video_queue_size: int = _VIDEO_QUEUE_SIZE,
        fade_s: float = _DEFAULT_FADE_S,
        turn_render_timeout: float = _DEFAULT_TURN_RENDER_TIMEOUT,
        audio_sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self._deque: deque[_FrameOrEnd] = deque()
        self._new_item = asyncio.Event()
        self._video_queue_size = video_queue_size
        self._video_count = 0
        self._dropped_video = 0
        self._fade_s = fade_s
        self._turn_render_timeout = turn_render_timeout
        self._audio_sample_rate = audio_sample_rate
        self._format_mismatches = 0

        self._first_video_frame: asyncio.Future[STVVideoFrame] | None = None
        self._geometry: tuple[int, int] | None = None
        self._geometry_mismatches = 0

        # segment protocol
        self._segment_open = False  # real audio forwarded since the last close
        self._input_closed = True  # the generator reports the input side
        self._output_drained = False  # stop edge seen while the input was open
        self._input_had_audio = False  # real audio pushed in this input segment

        # mute window (barge-in)
        self._muting = False
        self._mute_deadline = 0.0

        # clear-pending race guard
        self._clear_pending = False
        self._clear_done = asyncio.Event()
        self._clear_done.set()

        # liveness: fresh SERVER frames only. The playback loop keeps synthesizing
        # sink writes forever after a dead transport, so writes prove nothing.
        self._last_server_frame_time = time.monotonic()
        self._render_deadline: float | None = None
        self._rendered_turns = 0  # any real audio played; resets the stall streak

    # --- STVOutput (called from the SDK's playback loop; must not block) ---

    async def write_audio(self, frame: STVAudioFrame) -> None:
        if is_silence(frame):
            # Checked first: the fill emitted before the first turn is 16 kHz
            # shaped, so it would otherwise look like a format fault every session.
            return

        if frame.sample_rate != self._audio_sample_rate or frame.num_channels != NUM_CHANNELS:
            # The room's audio track is fixed at the format chosen when the runner
            # was built, and the source rejects anything else - which would kill
            # the runner's forwarding loop for the rest of the session. Ojin echoes
            # back whatever format it was fed, so this should not happen; drop the
            # frame rather than take the session down with it.
            self._format_mismatches += 1
            if self._format_mismatches == 1:
                logger.error(
                    "ojin returned audio in an unexpected format; dropping it",
                    extra={
                        "expected": (self._audio_sample_rate, NUM_CHANNELS),
                        "got": (frame.sample_rate, frame.num_channels),
                    },
                )
            return

        if is_silence(frame):
            return
        if self._muting:
            if time.monotonic() < self._mute_deadline:
                return
            self._muting = False  # timer fallback: the fade must be over by now

        self._segment_open = True
        self._render_deadline = None  # the turn is rendering
        self._rendered_turns += 1
        self._append(to_audio_frame(frame))

    async def write_video(self, frame: STVVideoFrame) -> None:
        if frame.source_bytes:
            self._last_server_frame_time = time.monotonic()

        if self._muting and frame.frame_type == FrameType.START_OF_SPEECH:
            # The replacement turn's first frame. Video precedes audio within a
            # tick, so unmuting here never eats the new turn's opening chunk.
            self._muting = False

        first = self._first_video_frame_fut()
        if not first.done():
            first.set_result(frame)

        if self._geometry is None:
            self._geometry = (frame.width, frame.height)
        elif (frame.width, frame.height) != self._geometry:
            # rtc.VideoSource is fixed at the first frame's size; a mismatched
            # frame would corrupt the track. Drop it, keep the stream alive.
            self._geometry_mismatches += 1
            if self._geometry_mismatches == 1:
                logger.error(
                    "ojin frame geometry changed mid-session; dropping mismatched frames",
                    extra={"expected": self._geometry, "got": (frame.width, frame.height)},
                )
            return

        video_frame = to_video_frame(frame)
        if video_frame is not None:
            self._append(video_frame, is_video=True)

    def on_event(self, event: STVEvent, **kwargs: object) -> None:
        # The WebSocket client never calls this; lifecycle arrives via add_listener.
        pass

    # --- segment protocol (the input side is reported by the generator) ---

    def note_input_segment_open(self) -> None:
        self._input_closed = False
        self._output_drained = False
        self._input_had_audio = False
        self._render_deadline = None

    def note_input_audio(self) -> None:
        """A real chunk was pushed; its echo has not played yet.

        Clearing the drained flag here rather than on output ticks stops a
        back-to-back final-chunk + input-close from emitting the marker while the
        refilled tail is still unplayed.
        """
        self._output_drained = False
        self._input_had_audio = True

    def note_input_segment_end(self, *, had_real_audio: bool = True) -> None:
        self._input_closed = True

        if not had_real_audio and not self._segment_open:
            # An all-zero utterance never echoes back, so no stop edge will ever
            # close it — but the runner captured the segment and the session is
            # waiting on its report. Emit now.
            self._append(AudioSegmentEnd())
            return

        if self._segment_open and self._output_drained:
            self._emit_segment_end()
            return

        if self._input_had_audio and not self._segment_open:
            # Fed but not rendering yet: bound the wait. A healthy transport can
            # stream idle frames forever while a turn never renders.
            self._render_deadline = time.monotonic() + self._turn_render_timeout

    def on_bot_stopped_speaking(self, **kwargs: object) -> None:
        if self._muting:
            # NOT a fade-end signal: the speaking predicate includes
            # `not interrupted`, so this fires one tick after interrupt() while
            # near-full-gain fade chunks are still coming. Unmuting here would
            # leak the audible fade.
            return
        if not self._segment_open:
            return
        if self._input_closed:
            self._emit_segment_end()
        else:
            # A mid-turn underrun: TTS is slower than playback. Not a turn end.
            self._output_drained = True

    def _emit_segment_end(self) -> None:
        self._segment_open = False
        self._output_drained = False
        self._render_deadline = None
        self._append(AudioSegmentEnd())

    # --- barge-in ---

    def note_clear_pending(self) -> None:
        """Set synchronously from the audio receiver's clear_buffer dispatch.

        The runner schedules its interruption as a task, so without this a fast
        replacement turn could open before the interrupt lands and have its own
        buffers cleared by it.
        """
        self._clear_pending = True
        self._clear_done.clear()

    def begin_clear(self) -> None:
        """Purge queued audio and markers, keep video, mute until an unmute trigger.

        Markers are purged because the runner tracks only an ``_audio_playing``
        flag: its interrupted report at clear balances the io-side count, while a
        retained marker consumed after a fast replacement turn started would end
        that new segment instantly.
        """
        kept = [f for f in self._deque if isinstance(f, rtc.VideoFrame)]
        self._deque.clear()
        self._deque.extend(kept)
        self._video_count = len(kept)
        self._segment_open = False
        self._output_drained = False
        self._input_closed = True
        self._input_had_audio = False
        self._render_deadline = None
        self._muting = True
        self._mute_deadline = time.monotonic() + self._fade_s + 0.25

    def abort_clear(self) -> None:
        """Lift the mute armed by :meth:`begin_clear` when no cancel was sent.

        A failed ``interrupt()`` means no fade is coming, and the surviving reply
        can swap in on an unmarked plain-speech frame — so no START_OF_SPEECH
        unmute would arrive and the timer window would eat its opening audio.
        """
        self._muting = False

    def finish_clear(self) -> None:
        self._clear_pending = False
        self._clear_done.set()

    @property
    def clear_pending(self) -> bool:
        return self._clear_pending

    async def wait_clear_done(self) -> None:
        await self._clear_done.wait()

    # --- discovery and liveness ---

    def _first_video_frame_fut(self) -> asyncio.Future[STVVideoFrame]:
        # Created lazily so the sink can be constructed outside a running loop
        # (a Future binds to its loop at construction). Production always builds
        # it inside start(); the unit tests do not.
        if self._first_video_frame is None:
            self._first_video_frame = asyncio.get_event_loop().create_future()
        return self._first_video_frame

    async def wait_for_first_video_frame(self, timeout: float) -> STVVideoFrame:
        return await asyncio.wait_for(asyncio.shield(self._first_video_frame_fut()), timeout)

    @property
    def last_server_frame_time(self) -> float:
        return self._last_server_frame_time

    @property
    def dropped_video_frames(self) -> int:
        return self._dropped_video

    @property
    def geometry_mismatches(self) -> int:
        return self._geometry_mismatches

    @property
    def format_mismatches(self) -> int:
        return self._format_mismatches

    @property
    def rendered_turns(self) -> int:
        return self._rendered_turns

    def render_deadline_expired(self) -> bool:
        return self._render_deadline is not None and time.monotonic() > self._render_deadline

    def force_segment_end(self) -> None:
        """Close a fed turn the server never rendered, so the session can proceed."""
        self._render_deadline = None
        self._segment_open = False
        self._output_drained = False
        self._append(AudioSegmentEnd())

    # --- consumer side ---

    @property
    def pending(self) -> tuple[_FrameOrEnd, ...]:
        """Everything queued but not yet consumed.

        next_frame() blocks when the queue is empty, so this is the only way to
        observe that nothing was emitted - which is most of what the demux rules
        are about.
        """
        return tuple(self._deque)

    async def next_frame(self) -> _FrameOrEnd:
        while not self._deque:
            self._new_item.clear()
            await self._new_item.wait()

        item = self._deque.popleft()
        if isinstance(item, rtc.VideoFrame):
            self._video_count -= 1
        return item

    def _append(self, item: _FrameOrEnd, *, is_video: bool = False) -> None:
        if is_video:
            if self._video_count >= self._video_queue_size:
                for i, queued in enumerate(self._deque):
                    if isinstance(queued, rtc.VideoFrame):
                        del self._deque[i]
                        self._video_count -= 1
                        self._dropped_video += 1
                        if self._dropped_video % self._video_queue_size == 1:
                            logger.warning(
                                "avatar video queue full, dropping oldest frames",
                                extra={"dropped_video_frames": self._dropped_video},
                            )
                        break
            self._video_count += 1

        self._deque.append(item)
        self._new_item.set()


class OjinVideoGenerator(VideoGenerator):
    """Adapts Ojin's push-based continuous stream to the runner's pull interface.

    Exactly one ``start_turn()`` per utterance, and the input side (open, real
    audio, close) is reported to the sink, whose segment-end rule needs it.
    """

    def __init__(self, client: OjinSTVClient, sink: _FrameSink) -> None:
        self._client = client
        self._sink = sink
        self._turn_started = False
        self._turn_had_real_audio = False
        self._gap_interrupt_logged = False
        self._last_interrupt_true = 0.0
        self._fade_s = sink._fade_s
        self._sample_rate = sink._audio_sample_rate
        self._resampler: rtc.AudioResampler | None = None
        self._resampler_input_rate = 0

    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            self._turn_started = False
            self._sink.note_input_segment_end(had_real_audio=self._turn_had_real_audio)
            return

        if not self._turn_started:
            if self._sink.clear_pending:
                # A barge-in is in flight; opening the turn first would let the
                # interrupt clear this new turn's buffers.
                await self._sink.wait_clear_done()

            # Set before the await: start_turn() suspends on a real websocket send
            # and anything running in that window must already see the open turn.
            self._turn_started = True
            self._turn_had_real_audio = False
            # Opened before the await for the same reason the flag is set before
            # it: anything running during the suspension must see the segment as
            # already open. Opening it afterwards would reset the sink's
            # input-side state and discard audio such a caller had reported,
            # leaving an unrenderable turn with no deadline to close it.
            self._sink.note_input_segment_open()
            try:
                await self._client.start_turn()
            except Exception:
                self._turn_started = False
                # Never raise: an escape kills the runner's _read_audio task for
                # the rest of the session (it is only guarded by log_exceptions).
                logger.exception("ojin start_turn failed; dropping this chunk")
                return

        try:
            pcm = downmix_to_mono(bytes(frame.data), frame.num_channels)
            if frame.sample_rate != self._sample_rate:
                # Ojin plays back whatever we feed it, and the room's track runs at
                # a fixed rate, so anything else has to be converted here. The
                # framework installs its own resampler only on a segment's first
                # frame, so an off-rate frame can still reach us mid-stream.
                pcm = self._resample(pcm, frame.sample_rate)
        except Exception:
            logger.exception("ojin audio conversion failed; dropping this chunk")
            return

        if not pcm:
            return

        if pcm.strip(b"\x00"):
            # Only real audio counts: the SDK discards all-zero payloads of about
            # half a second without buffering them, and even a buffered zero chunk
            # echoes back as zeros that the sink's silence gate drops - so a zero
            # chunk must never satisfy the segment-end rule.
            self._turn_had_real_audio = True
            self._sink.note_input_audio()

        try:
            await self._client.send_tts_audio(pcm, self._sample_rate, NUM_CHANNELS)
        except Exception:
            logger.exception("ojin send_tts_audio failed; dropping this chunk")

    def _resample(self, pcm: bytes, input_rate: int) -> bytes:
        if self._resampler is None or self._resampler_input_rate != input_rate:
            logger.debug(
                "resampling avatar input audio",
                extra={"from": input_rate, "to": self._sample_rate},
            )
            self._resampler = rtc.AudioResampler(
                input_rate=input_rate, output_rate=self._sample_rate, num_channels=NUM_CHANNELS
            )
            self._resampler_input_rate = input_rate

        frame = rtc.AudioFrame(
            data=pcm,
            sample_rate=input_rate,
            num_channels=NUM_CHANNELS,
            samples_per_channel=len(pcm) // 2,
        )
        return b"".join(bytes(out.data) for out in self._resampler.push(frame))

    async def clear_buffer(self) -> None:
        """Barge-in. Never raises.

        The runner's clear handler swallows exceptions and would then skip its
        interrupted playback-finished report, wedging the session; this also runs
        against an already-closed client after a fatal-error teardown.
        """
        turn_was_open = self._turn_started
        self._turn_started = False
        self._sink.begin_clear()
        try:
            interrupted = await self._client.interrupt()
            if interrupted:
                self._last_interrupt_true = time.monotonic()
            elif time.monotonic() - self._last_interrupt_true < self._fade_s:
                # Our own recent interrupt is still fading out. Whatever the SDK's
                # reason for refusing this one, keeping the mute is what matters:
                # lifting it would leak that fade into the room.
                logger.debug("ojin barge-in refused while a fade is in flight")
            else:
                # Nothing was cancelled and no fade is coming, so the mute is pure
                # harm: whatever plays next would lose its opening audio.
                self._sink.abort_clear()
                # begin_clear() already retired this segment's input state, so a
                # turn that was not cancelled just renders into a fresh output
                # segment when its frames arrive.
                if turn_was_open:
                    if not self._gap_interrupt_logged:
                        # Documented v1 limitation: a turn that is already in
                        # flight cannot be cancelled.
                        self._gap_interrupt_logged = True
                        # The SDK reports only "refused", not why: an in-flight
                        # turn that has not started rendering (the inference gap)
                        # and a cancel still being acknowledged are
                        # indistinguishable from here. Either way this turn was
                        # not cancelled.
                        logger.warning(
                            "ojin refused a barge-in and the pending turn was not "
                            "cancelled, so that reply will play (SDK limitation)"
                        )
        except Exception:
            logger.exception("ojin interrupt failed; continuing teardown-safe")
        finally:
            self._sink.finish_clear()

    def __aiter__(self) -> AsyncIterator[_FrameOrEnd]:
        return self._stream_impl()

    async def _stream_impl(self) -> AsyncGenerator[_FrameOrEnd, None]:
        while True:
            yield await self._sink.next_frame()


def _build_avatar_options(
    first_frame: STVVideoFrame, *, audio_sample_rate: int, video_fps: int
) -> AvatarOptions:
    """Derive the runner's options from the first frame the model produced.

    Frame size belongs to the Ojin model and is not known before it sends one, so
    the runner is built after the stream opens rather than from a fixed default.
    """
    return AvatarOptions(
        video_width=first_frame.width,
        video_height=first_frame.height,
        video_fps=video_fps,
        audio_sample_rate=audio_sample_rate,
        audio_channels=NUM_CHANNELS,
    )


class AvatarSession(BaseAvatarSession):
    """An Ojin avatar session.

    Renders in process and publishes the avatar on the agent's own participant.
    """

    def __init__(
        self,
        *,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        config_id: NotGivenOr[str] = NOT_GIVEN,
        ws_url: NotGivenOr[str] = NOT_GIVEN,
        stv_config: NotGivenOr[STVConfig] = NOT_GIVEN,
        audio_sample_rate: int = SAMPLE_RATE,
        session_ready_timeout: float = _DEFAULT_SESSION_READY_TIMEOUT,
        first_frame_timeout: float = _DEFAULT_FIRST_FRAME_TIMEOUT,
        watchdog_timeout: float = _DEFAULT_WATCHDOG_TIMEOUT,
        turn_render_timeout: float = _DEFAULT_TURN_RENDER_TIMEOUT,
    ) -> None:
        super().__init__()

        api_key_val = api_key if utils.is_given(api_key) else os.getenv("OJIN_API_KEY")
        config_id_val = config_id if utils.is_given(config_id) else os.getenv("OJIN_CONFIG_ID")
        if not api_key_val:
            raise OjinException("OJIN_API_KEY must be set by argument or environment variable")
        if not config_id_val:
            raise OjinException("OJIN_CONFIG_ID must be set by argument or environment variable")

        self._api_key = api_key_val
        self._config_id = config_id_val
        self._ws_url = ws_url if utils.is_given(ws_url) else os.getenv("OJIN_WS_URL")
        self._stv_config = stv_config if utils.is_given(stv_config) else None
        self._audio_sample_rate = audio_sample_rate
        self._session_ready_timeout = session_ready_timeout
        self._first_frame_timeout = first_frame_timeout
        self._watchdog_timeout = watchdog_timeout
        self._turn_render_timeout = turn_render_timeout

        self._client: OjinSTVClient | None = None
        self._sink: _FrameSink | None = None
        self._audio_output: QueueAudioOutput | None = None
        self._avatar_runner: AvatarRunner | None = None

        self._ready = asyncio.Event()
        self._fatal_error: str | None = None
        self._closed = False
        self._render_stalls = 0
        self._last_rendered_turns = 0
        self._degrade_starts = 0

        self._watchdog_task: asyncio.Task[None] | None = None
        # _degrade calls aclose(), so its own handle must never be among the
        # tasks aclose cancels - that would cancel its caller mid-teardown.
        self._degrade_task: asyncio.Task[None] | None = None
        self._null_drain_task: asyncio.Task[None] | None = None

    @property
    def avatar_identity(self) -> str:
        # The avatar publishes on the agent's own participant, so the identity is
        # the local one.
        if self._room is not None:
            return self._room.local_participant.identity
        return "ojin-avatar"

    @property
    def provider(self) -> str:
        return "ojin"

    def _effective_fps(self) -> int:
        return int(self._stv_config.fps) if self._stv_config is not None else VIDEO_FPS

    def _effective_fade_s(self) -> float:
        if self._stv_config is not None:
            return float(self._stv_config.interrupt_audio_fade_s)
        return _DEFAULT_FADE_S

    def _build_sink(self) -> _FrameSink:
        return _FrameSink(
            fade_s=self._effective_fade_s(),
            turn_render_timeout=self._turn_render_timeout,
            audio_sample_rate=self._audio_sample_rate,
        )

    def _wire_listeners(self, client: OjinSTVClient, sink: _FrameSink) -> None:
        """Register lifecycle handlers.

        Every handler takes the SDK's kwargs. The emitter calls handlers as
        ``cb(**kwargs)`` and turns a signature mismatch into a logged exception
        under its own logger rather than an error here, so a wrong signature
        silently does nothing: SESSION_READY carries ``session_data``, and a
        server-side ERROR carries ``code`` on top of ``message``/``fatal``.
        """
        client.add_listener(STVEvent.SESSION_READY, lambda **_: self._ready.set())
        client.add_listener(STVEvent.ERROR, self._on_error)
        client.add_listener(STVEvent.CLOSED, self._on_closed)
        client.add_listener(STVEvent.BOT_STOPPED_SPEAKING, sink.on_bot_stopped_speaking)

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        await super().start(agent_session, room)
        try:
            await self._start(agent_session, room)
        except BaseException:
            # Every failure path - cancellation included - rolls back what start()
            # installed, the base session's listeners and join task among it.
            await self.aclose()
            raise

    async def _start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        sink = self._build_sink()
        self._sink = sink

        client_kwargs: dict[str, object] = {}
        if self._ws_url:
            client_kwargs["ws_url"] = self._ws_url
        if self._stv_config is not None:
            client_kwargs["config"] = self._stv_config

        client = OjinSTVClient(
            api_key=self._api_key,
            config_id=self._config_id,
            output=sink,
            **client_kwargs,
        )
        self._client = client
        self._wire_listeners(client, sink)

        # start() does not raise on connect exhaustion: it emits a fatal ERROR and
        # returns, so the ready gate below is what turns that into an exception.
        await client.start()
        try:
            await asyncio.wait_for(self._ready.wait(), self._session_ready_timeout)
        except asyncio.TimeoutError as e:
            raise OjinException(
                f"ojin session was not ready within {self._session_ready_timeout}s"
            ) from e

        if self._fatal_error is not None:
            raise OjinException(f"ojin session failed to start: {self._fatal_error}")

        try:
            first_frame = await sink.wait_for_first_video_frame(self._first_frame_timeout)
        except asyncio.TimeoutError as e:
            raise OjinException(
                f"ojin sent no video frame within {self._first_frame_timeout}s"
            ) from e

        options = _build_avatar_options(
            first_frame,
            audio_sample_rate=self._audio_sample_rate,
            video_fps=self._effective_fps(),
        )
        logger.debug(
            "starting ojin avatar runner",
            extra={"width": options.video_width, "height": options.video_height},
        )

        audio_output = QueueAudioOutput(
            sample_rate=self._audio_sample_rate, wait_playback_start=True
        )
        # Synchronous dispatch: this is what lets a new turn see a barge-in that
        # the runner has only scheduled.
        receiver: AudioReceiver = audio_output
        receiver.on("clear_buffer", lambda *_: sink.note_clear_pending())
        self._audio_output = audio_output

        self._avatar_runner = AvatarRunner(
            room=room,
            video_gen=OjinVideoGenerator(client, sink),
            audio_recv=audio_output,
            options=options,
        )
        await self._avatar_runner.start()

        agent_session.output.audio = audio_output
        self._start_watchdog()

    # --- liveness ---

    def _start_watchdog(self) -> None:
        self._watchdog_task = asyncio.create_task(self._watchdog())

    async def _watchdog(self) -> None:
        while not self._closed:
            await asyncio.sleep(max(self._watchdog_timeout / 2, 0.01))
            if self._sink is None or self._closed:
                return

            idle = time.monotonic() - self._sink.last_server_frame_time
            if idle > self._watchdog_timeout:
                logger.error(
                    "no fresh frames from ojin; treating the session as dead",
                    extra={"idle_seconds": round(idle, 1)},
                )
                self._on_fatal(f"no frames from ojin for {idle:.1f}s")
                return

            self._check_render_liveness()

    def _check_render_liveness(self) -> None:
        """Bound a turn that was fed but never rendered.

        A healthy transport can stream idle frames forever while a turn never
        produces speech: no error, no close, no stop edge, and the frame watchdog
        stays silent while the session waits on a marker that will never come.
        """
        sink = self._sink
        if sink is None or not sink.render_deadline_expired():
            return

        if sink.rendered_turns != self._last_rendered_turns:
            # A turn rendered since the previous stall, so the streak is broken.
            self._last_rendered_turns = sink.rendered_turns
            self._render_stalls = 0

        self._render_stalls += 1
        logger.error(
            "ojin accepted a turn but never rendered it; closing the segment",
            extra={"consecutive_stalls": self._render_stalls},
        )
        sink.force_segment_end()
        if self._render_stalls >= 2:
            self._on_fatal("ojin repeatedly failed to render a turn")

    # --- failure handling ---

    def _on_error(self, message: str = "", fatal: bool = False, **kwargs: object) -> None:
        if not fatal:
            # No SDK path emits this today; handled defensively.
            logger.warning("ojin reported an error", extra={"ojin_error": message})
            return
        self._on_fatal(message)

    def _on_closed(self, **kwargs: object) -> None:
        if self._closed:
            return  # our own teardown
        self._on_fatal("ojin client closed unexpectedly")

    def _on_fatal(self, message: str) -> None:
        if self._fatal_error is not None:
            # First wins. The SDK emits ERROR(fatal) and then CLOSED for every
            # server-side fatal, and a second degrade would race a second
            # null-drain onto the same channel.
            return

        self._fatal_error = message
        logger.error(
            "ojin avatar session failed; tearing down the avatar",
            extra={"ojin_error": message},
        )
        self._ready.set()  # unblock a start() still waiting

        if self._avatar_runner is not None and self._degrade_task is None:
            self._degrade_task = asyncio.create_task(self._degrade())

    async def _degrade(self) -> None:
        """Tear the avatar down without wedging the agent session.

        agent_session.output.audio still points at our QueueAudioOutput, and
        closing the runner leaves nobody draining it - the next utterance would
        block forever on wait_for_playout. The null drain keeps the playback
        reports flowing so the session degrades (silently, since the avatar
        carried the only audio track) instead of deadlocking.
        """
        self._degrade_starts += 1
        await self.aclose()

        if self._audio_output is not None and self._null_drain_task is None:
            self._null_drain_task = asyncio.create_task(self._null_drain(self._audio_output))

    async def _null_drain(self, audio_output: QueueAudioOutput) -> None:
        position = 0.0
        started = False

        def on_clear(*_: object) -> None:
            # The runner's stale clear handler cannot be relied on: its
            # _audio_playing gate froze at teardown (typically False at idle).
            nonlocal position, started
            audio_output.notify_playback_finished(playback_position=position, interrupted=True)
            position, started = 0.0, False

        receiver: AudioReceiver = audio_output
        receiver.on("clear_buffer", on_clear)

        async for frame in audio_output:
            if isinstance(frame, AudioSegmentEnd):
                audio_output.notify_playback_finished(playback_position=position, interrupted=False)
                position, started = 0.0, False
                continue

            if not started:
                started = True
                audio_output.notify_playback_started()
            position += frame.duration

    async def aclose(self) -> None:
        if self._closed:
            # Teardown already ran, but a degraded session starts its null drain
            # *after* that, so the second call (typically the job's shutdown
            # callback) is what stops it. Without this a long-lived worker leaks
            # one drain task per degraded job.
            await self._stop_null_drain()
            return
        self._closed = True

        await super().aclose()

        if self._watchdog_task is not None:
            await utils.aio.cancel_and_wait(self._watchdog_task)
            self._watchdog_task = None
        if self._avatar_runner is not None:
            await self._avatar_runner.aclose()
            self._avatar_runner = None

        if self._client is not None:
            # Safe against a double close: the SDK closes itself on server fatals.
            await self._client.close()
            self._client = None

        await self._stop_null_drain()

    async def _stop_null_drain(self) -> None:
        if self._null_drain_task is not None:
            await utils.aio.cancel_and_wait(self._null_drain_task)
            self._null_drain_task = None
