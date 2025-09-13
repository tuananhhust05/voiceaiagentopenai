import os
import json
import base64
import asyncio
import websockets
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.websockets import WebSocketDisconnect
from twilio.twiml.voice_response import VoiceResponse, Connect, Say, Stream
from dotenv import load_dotenv
import wave
from fastapi.responses import FileResponse
import edge_tts
from pydub import AudioSegment
import webrtcvad
import soundfile as sf
import requests
import audioop
import traceback
import numpy as np
from faster_whisper import WhisperModel
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/your_tts")
load_dotenv()

# ==== Faster-Whisper model ====
from faster_whisper import WhisperModel
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
speaker_embedding = tts.synthesizer.tts_model.speaker_manager.compute_embedding_from_clip("customvoice.wav")
name = "andrea"
tts.synthesizer.tts_model.speaker_manager.name_to_id[name] = 0
tts.synthesizer.tts_model.speaker_manager.embeddings_by_names[name] = [speaker_embedding] 

# ==== Global Config ====
vad = webrtcvad.Vad(0)  # 0 = nhạy thấp, 3 = nhạy cao
frame_duration_ms = 30
sample_rate = 8000
frame_bytes = int(sample_rate * 2 * frame_duration_ms / 1000)  # 16-bit PCM → 2 bytes
buffer_pcm = b""
speech_buffer = b""
is_processing = False
stream_sid = None
current_websocket = None
interrupt = False
hangover_frames = 10
VOICE = "en-US-AriaNeural"  # giọng của edge-tts


# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PORT = int(os.getenv('PORT', 5059))
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.8))
SYSTEM_MESSAGE = (
    "You are a helpful and bubbly AI assistant who loves to chat about "
    "anything the user is interested in and is prepared to offer them facts. "
    "You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. "
    "Always stay positive, but work in a joke when appropriate."
)
VOICE = 'alloy'
LOG_EVENT_TYPES = [
    'error', 'response.content.done', 'rate_limits.updated',
    'response.done', 'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped', 'input_audio_buffer.speech_started',
    'session.created', 'session.updated'
]
SHOW_TIMING_MATH = False

app = FastAPI()

if not OPENAI_API_KEY:
    raise ValueError('Missing the OpenAI API key. Please set it in the .env file.')

@app.get("/", response_class=JSONResponse)
async def index_page():
    return {"message": "Twilio Media Stream Server is running!"}

@app.get("/download-audio/{filename}")
async def download_audio(filename: str):
    file_path = os.path.join(".", filename)
    if not os.path.exists(file_path):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    return FileResponse(path=file_path, filename=filename, media_type="audio/wav")

@app.api_route("/incoming-call", methods=["GET", "POST"])
async def handle_incoming_call(request: Request):
    """Handle incoming call and return TwiML response to connect to Media Stream."""
    response = VoiceResponse()
    # <Say> punctuation to improve text-to-speech flow
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

# --- Generate speech with edge-tts ---
async def generate_tts_wav(text: str, output_file: str):
    communicate = edge_tts.Communicate(text, voice="en-US-AriaNeural")  # English female voice
    await communicate.save(output_file)

# --- Convert to Twilio µ-law 8kHz mono WAV ---
def convert_to_twilio_format(input_file: str, output_file: str):
    audio = AudioSegment.from_file(input_file)
    audio = audio.set_frame_rate(8000).set_channels(1)
    audio.export(output_file, format="wav", codec="pcm_mulaw")
    return output_file

@app.websocket("/media-stream")
async def handle_media_stream_from_file(websocket: WebSocket):
    global buffer_pcm, speech_buffer, stream_sid, interrupt,hangover_frames
    print("Client connected")
    await websocket.accept()
    async for message in websocket.iter_text():
        try:
            data = json.loads(message)
        except Exception as e:
            print("❌ JSON parse error:", e)
            continue
        event = data.get("event")
        if event == "start":
            stream_sid = data["start"]["streamSid"]
            print(f"Incoming stream started: {stream_sid}")
        elif event == "media":
            payload_b64 = data["media"]["payload"]
            ulaw_bytes = base64.b64decode(payload_b64)
            pcm16_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
            buffer_pcm += pcm16_bytes
            silence_counter = 0
            while len(buffer_pcm) >= frame_bytes:
                frame = buffer_pcm[:frame_bytes]
                buffer_pcm = buffer_pcm[frame_bytes:]

                is_speech = vad.is_speech(frame, sample_rate)
                if is_speech:
                    interrupt = True
                    silence_counter = 0
                    speech_buffer += frame
                else:
                    silence_counter += 1
                    interrupt = False
                    print("Silence counter:", silence_counter)
                    if(silence_counter > 2):
                        if len(speech_buffer) > 0:
                                if (interrupt == False): 
                                    try:
                                        llm_response = await transcribe_and_respond(speech_buffer)
                                        speech_buffer = b""
                                        print("llm response .....})))", llm_response)
                                        if llm_response :
                                            """
                                            Instead of reading an existing file, we dynamically create one
                                            with edge-tts, convert it, and stream it back to Twilio.
                                            """
                                            print("Start create file")
                                            wav = tts.synthesizer.tts(
                                                text=llm_response,
                                                speaker_name=name,
                                                language_name="en"
                                            )
                                            tts.synthesizer.save_wav(wav, "edge_temp.wav")
                                            raw_file = "edge_temp.wav"
                                            # await generate_tts_wav(llm_response, raw_file)
                                            twilio_file = "edge_twilio.wav"
                                            convert_to_twilio_format(raw_file, twilio_file)
                                            file_path = twilio_file
                                            with open(file_path, "rb") as f:
                                                audio_data = f.read()  
                                            chunk_size = 160
                                            try:  
                                                print("Start send ...")
                                                for i in range(0, len(audio_data), chunk_size):
                                                    if (interrupt == False): 
                                                        chunk = audio_data[i:i+chunk_size]
                                                        audio_payload = base64.b64encode(chunk).decode('utf-8')
                                                        audio_delta = {
                                                            "event": "media",
                                                            "streamSid": stream_sid,
                                                            "media": {"payload": audio_payload}
                                                        }
                                                        await websocket.send_json(audio_delta)
                                                await websocket.send_json({
                                                    "event": "stop",
                                                    "streamSid": stream_sid
                                                })
                                            except Exception as e:
                                                print(f"Error: {e}")
                                    except Exception as e:
                                        traceback.print_exc()
                                interrupt = False
                        interrupt = False

# ==== TRANSCRIBE + CALL LLM ====
async def transcribe_and_respond(pcm_bytes):
    global is_processing
    if is_processing:
        print("⏳ waiting for previous transcription to finish...")
        return None

    # convert cho Whisper
    audio_np = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    sf.write("temp.wav", audio_np, sample_rate)

    segments, _ = model.transcribe("temp.wav", beam_size=1)
    text = "".join([seg.text for seg in segments])
    print("📝 Transcript:", text)
    if not text:
        return None

    is_processing = True
    llm_response = "Please repeat that."

    try:
        
        payload = {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "0",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "83868",
                                "phone_number_id": "123456123"
                            },
                            "contacts": [
                                {
                                    "profile": {
                                        "name": "test user name"
                                    },
                                    "wa_id": "16315558881180"
                                }
                            ],
                            "messages": [
                                {
                                    "from": "16315551180",
                                    "id": "ABGGFlA5Fpa",
                                    "timestamp": "1504902988",
                                    "type": "text",
                                    "text": {
                                        "body": text.strip()
                                    }
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }
        response = requests.post(
            "http://127.0.0.1:8501/webhook",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        print("Webhook response status:", response.json())
        response.raise_for_status()
        llm_response = response.json().get("reply", "Please repeat that.")
    except Exception as e:
        print("❌ Webhook error:", e)

    print("🤖 LLM Response:", llm_response)
    is_processing = False
    return llm_response


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