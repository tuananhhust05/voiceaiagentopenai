import os
import json
import base64
import asyncio
import websockets
import uuid
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
import wave
import httpx
import glob
from datetime import datetime
import audioop

load_dotenv()

# Configuration
ELEVEN_API_KEY = os.getenv('ELEVENLABS_API_KEY')
AGENT_ID = os.getenv('ELEVENLABS_AGENT_ID')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5059))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))
SYSTEM_MESSAGE = (
    "You are a friendly, expressive AI voice assistant. "
    "When you speak, sound natural and human-like: use short sentences, "
    "pauses, and conversational rhythm. Emphasize important words, vary "
    "your tone to avoid monotony, and sometimes add small hesitations like "
    "'well,' or 'you know' to feel more real. "
    "Speak like you’re chatting with a friend — warm, engaging, and clear. "
    "When telling a joke or fun fact, use extra intonation and timing for effect. "
    "Avoid sounding robotic or like you’re reading text — always imagine you are "
    "telling a story out loud. "
    "Always reply in Italian, regardless of the user's language."
)

VOICE = 'coral'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created', 'session.updated'
]
SHOW_TIMING_MATH = False

app = FastAPI()

# Create uploads folder
UPLOAD_DIR = "uploads/audios"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files
app.mount("/uploads/audios", StaticFiles(directory=UPLOAD_DIR), name="uploads/audios")

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    response.say(
        "Please wait while we connect your call to the A. I. voice assistant, powered by Twilio and the Open A I Realtime API",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    response.pause(length=1)
    response.say(
        "O.K. you can start talking!",
        voice="Google.en-US-Chirp3-HD-Aoede"
    )
    host = request.url.hostname
    connect = Connect()
    connect.stream(url=f'wss://{host}/media-stream')
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

@app.post("/outbound-twiml")
async def outbound_twiml(request: Request):
    response = VoiceResponse()
    response.say("Hello! I am supporter from 4skale , how can I help you ?")
    connect = Connect()
    connect.stream(url=f"wss://4skale.com/media-stream")
    response.append(connect)
    return HTMLResponse(content=str(response), media_type="application/xml")

# Hàm lấy file mới nhất trong folder
def get_latest_file(folder: str):
    list_of_files = glob.glob(os.path.join(folder, "*"))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

@app.post("/twilio/status-callback")
async def twilio_status_callback(request: Request):
    form_data = await request.form()
    call_status = form_data.get("CallStatus")
    duration = form_data.get("CallDuration")
    recording_url = form_data.get("RecordingUrl")

    latest_file = get_latest_file(UPLOAD_DIR)

    if not latest_file:
        return JSONResponse({"error": "No file found in uploads/twilio"}, status_code=404)
    
    payload = {
        "duration": duration or 0,
        "recording_url": recording_url or f"/{latest_file}",
        "transcript": "Transcripting.",  
        "sentiment": "positive",
        "sentiment_score": 0.85,
        "status": call_status or "completed",
        "local_file": latest_file,
        "received_at": datetime.utcnow().isoformat()
    }

    
    async def call_webhook():
        try:
            async with httpx.AsyncClient() as client:
                r = await client.put(
                    "https://4skale.com/api/webhook/auto-update-latest",
                    json=payload,
                    timeout=10.0
                )
                print("Webhook response:", r.status_code, r.text)
        except Exception as e:
            print("Failed to call webhook:", e)

    await call_webhook()

    return JSONResponse({"message": "Callback processed", "payload": payload})

@app.websocket("/media-stream")
async def handle_media_stream(websocket: WebSocket):
    """Proxy WebSocket giữa Twilio ↔ ElevenLabs Agent"""
    await websocket.accept()
    print("✅ Twilio connected")

    # Lưu audio ghi âm
    filename = f"{uuid.uuid4()}.wav"
    filepath = os.path.join(UPLOAD_DIR, filename)
    wave_file = wave.open(filepath, "wb")
    wave_file.setnchannels(1)
    wave_file.setsampwidth(2)  # 16-bit PCM
    wave_file.setframerate(8000)

    try:
        # --- Signed URL của ElevenLabs ---
        signed_url = "wss://api.elevenlabs.io/v1/convai/conversation?agent_id=agent_7601k5tn5fffe65a7wjsg6tfd32z&conversation_signature=cvtkn_1601k5v68h0necjtbpae2b3w42wj"  # set từ .env
        print("✅ Connecting to ElevenLabs Agent:", signed_url)

        async with websockets.connect(signed_url) as eleven_ws:
            print("✅ Connected to ElevenLabs Agent")

            # --- Khởi tạo session realtime ---
            session_update = {
                "type": "session.update",
                "session": {
                    "type": "realtime",
                    "model": "gpt-realtime",
                    "output_modalities": ["audio"],
                    "audio": {
                        "input": {"format": {"type": "audio/pcm"}},
                        "output": {"format": {"type": "audio/pcm"}, "voice": VOICE}
                    },
                    "instructions": SYSTEM_MESSAGE,
                }
            }
            await eleven_ws.send(json.dumps(session_update))
            print("📄 Session update sent to ElevenLabs")

            # --- Optional: AI nói trước ---
            initial_message = {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Ciao! Come stai?"}]
                }
            }
            await eleven_ws.send(json.dumps(initial_message))
            await eleven_ws.send(json.dumps({"type": "response.create"}))
            print("💬 Initial message sent to ElevenLabs")

            # --- Nhận dữ liệu từ Twilio ---
            async def receive_from_twilio():
                try:
                    async for message in websocket.iter_text():
                        data = json.loads(message)
                        if data["event"] == "media":
                            audio_payload = data["media"]["payload"]  # base64 PCM
                            await eleven_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_payload
                            }))
                        elif data["event"] == "start":
                            print("📞 Twilio stream started")
                        elif data["event"] == "stop":
                            print("🛑 Twilio stream stopped")
                except WebSocketDisconnect:
                    print("❌ Twilio disconnected")

            # --- Nhận dữ liệu từ ElevenLabs ---
            async def receive_from_eleven():
                try:
                    async for msg in eleven_ws:
                        response = json.loads(msg)
                        if response.get("type") == "output_audio_buffer.delta":
                            audio_b64 = response["audio"]
                            raw_audio = base64.b64decode(audio_b64)
                            wave_file.writeframes(raw_audio)
                            await websocket.send_json({
                                "event": "media",
                                "streamSid": "xxx",  # Twilio stream ID
                                "media": {"payload": audio_b64}
                            })
                        elif response.get("type") == "message":
                            print("🗨️ Agent:", response.get("text"))
                        else:
                            # Debug các event khác
                            if SHOW_TIMING_MATH:
                                print("ElevenLabs Event:", response)
                except Exception as e:
                    print("Error from ElevenLabs:", e)

            await asyncio.gather(receive_from_twilio(), receive_from_eleven())

    except Exception as e:
        print("❌ Error in media stream:", e)

    finally:
        wave_file.close()
        print(f"Call ended. Audio saved: {filepath}")

        # --- Tính duration ---
        try:
            with wave.open(filepath, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = round(frames / float(rate), 2)
        except Exception:
            duration = 0

        # --- Tạo recording_url ---
        recording_url = f"https://4skale.com/uploads/audios/{filename}"

        # --- Call webhook ---
        payload = {
            "duration": duration,
            "recording_url": recording_url,
            "transcript": "Transcripting.",  # TODO: thay bằng transcript thực tế
            "sentiment": "positive",
            "sentiment_score": 0.85,
            "status": "completed"
        }

        async def call_webhook():
            try:
                async with httpx.AsyncClient() as client:
                    r = await client.put(
                        "https://4skale.com/api/webhook/auto-update-latest",
                        json=payload,
                        timeout=10.0
                    )
                    print("Webhook response:", r.status_code, r.text)
            except Exception as e:
                print("Failed to call webhook:", e)

        asyncio.create_task(call_webhook())

async def send_initial_conversation_item(openai_ws):
    """Send initial conversation item if AI talks first."""
    initial_conversation_item = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Greet the user with 'Hello there! I am an AI voice assistant powered by Twilio and the OpenAI Realtime API. You can ask me for facts, jokes, or anything you can imagine. How can I help you?'"
                }
            ]
        }
    }
    await openai_ws.send(json.dumps(initial_conversation_item))
    await openai_ws.send(json.dumps({"type": "response.create"}))

async def initialize_session(openai_ws):
    """Control initial session with OpenAI."""
    session_update = {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "turn_detection": {"type": "server_vad"}
                },
                "output": {
                    "format": {"type": "audio/pcmu"},
                    "voice": VOICE
                }
            },
            "instructions": SYSTEM_MESSAGE,
        }
    }
    print('Sending session update:', json.dumps(session_update))
    await openai_ws.send(json.dumps(session_update))

    # Uncomment the next line to have the AI speak first
    # await send_initial_conversation_item(openai_ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)