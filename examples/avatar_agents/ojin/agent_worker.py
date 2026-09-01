import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, AgentSession, JobContext, cli
from livekit.plugins import ojin, openai

logger = logging.getLogger("ojin-avatar-example")
logger.setLevel(logging.INFO)

load_dotenv()

server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="alloy"),
    )

    # The plugin reads OJIN_API_KEY / OJIN_CONFIG_ID from the environment and
    # raises with a clear message when either is missing. The config id must name
    # a model with a live backend, otherwise the session fails with
    # BACKEND_UNAVAILABLE.
    ojin_avatar = ojin.AvatarSession()
    # Start the avatar before the session: it publishes the audio track the agent
    # speaks through, and the frame size comes from the model's first frame.
    await ojin_avatar.start(session, room=ctx.room)

    await session.start(
        agent=Agent(instructions="Talk to me!"),
        room=ctx.room,
    )
    session.generate_reply(instructions="say hello to the user")


if __name__ == "__main__":
    cli.run_app(server)
