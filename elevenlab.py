import os
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface
from dotenv import load_dotenv

load_dotenv()

# Lấy từ .env
API_KEY = os.getenv("ELEVENLABS_API_KEY")
AGENT_ID = os.getenv("AGENT_ID")

client = ElevenLabs(api_key=API_KEY)

conversation = Conversation(
    client=client,
    agent_id=AGENT_ID,
    requires_auth=True,
    audio_interface=DefaultAudioInterface(),  # Sử dụng micro + loa local để test
    callback_agent_response=lambda response: print(f"🤖 Agent: {response}"),
    callback_user_transcript=lambda transcript: print(f"🧑 You: {transcript}"),
)

conversation.start_session()
