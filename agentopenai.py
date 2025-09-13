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
import numpy as np
from faster_whisper import WhisperModel

load_dotenv()

# ==== Faster-Whisper model ====
from faster_whisper import WhisperModel
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

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
# async def handle_media_stream(websocket: WebSocket):
#     """Handle WebSocket connections between Twilio and OpenAI."""
#     print("Client connected")
#     wave_file = wave.open("openai_output.wav", "wb")
#     wave_file.setnchannels(1)
#     wave_file.setsampwidth(1)   # mu-law là 8bit
#     wave_file.setframerate(8000)
#     await websocket.accept()

#     async with websockets.connect(
#         f"wss://api.openai.com/v1/realtime?model=gpt-realtime&temperature={TEMPERATURE}",
#         additional_headers={
#             "Authorization": f"Bearer {OPENAI_API_KEY}"
#         }
#     ) as openai_ws:
#         await initialize_session(openai_ws)

#         # Connection specific state
#         stream_sid = None
#         latest_media_timestamp = 0
#         last_assistant_item = None
#         mark_queue = []
#         response_start_timestamp_twilio = None
        
#         async def receive_from_twilio():
#             """Receive audio data from Twilio and send it to the OpenAI Realtime API."""
#             nonlocal stream_sid, latest_media_timestamp
#             try:
#                 async for message in websocket.iter_text():
#                     data = json.loads(message)
#                     if data['event'] == 'media' and openai_ws.state.name == 'OPEN':
#                         latest_media_timestamp = int(data['media']['timestamp'])
#                         audio_append = {
#                             "type": "input_audio_buffer.append",
#                             "audio": data['media']['payload']
#                         }
#                         await openai_ws.send(json.dumps(audio_append))
#                     elif data['event'] == 'start':
#                         stream_sid = data['start']['streamSid']
#                         print(f"Incoming stream has started {stream_sid}")
#                         response_start_timestamp_twilio = None
#                         latest_media_timestamp = 0
#                         last_assistant_item = None
#                     elif data['event'] == 'mark':
#                         if mark_queue:
#                             mark_queue.pop(0)
#             except WebSocketDisconnect:
#                 print("Client disconnected.")
#                 if openai_ws.state.name == 'OPEN':
#                     await openai_ws.close()

#         async def send_to_twilio():
#             """Receive events from the OpenAI Realtime API, send audio back to Twilio."""
#             nonlocal stream_sid, last_assistant_item, response_start_timestamp_twilio
#             try:
#                 async for openai_message in openai_ws:
#                     response = json.loads(openai_message)
#                     if response['type'] in LOG_EVENT_TYPES:
#                         print(f"Received event: {response['type']}", response)

#                     if response.get('type') == 'response.output_audio.delta' and 'delta' in response:
#                         raw_audio = base64.b64decode(response['delta'])
#                         wave_file.writeframes(raw_audio)
#                         audio_payload = base64.b64encode(base64.b64decode(response['delta'])).decode('utf-8')
#                         audio_delta = {
#                             "event": "media",
#                             "streamSid": stream_sid,
#                             "media": {
#                                 "payload": audio_payload
#                             }
#                         }
#                         await websocket.send_json(audio_delta)


#                         if response.get("item_id") and response["item_id"] != last_assistant_item:
#                             response_start_timestamp_twilio = latest_media_timestamp
#                             last_assistant_item = response["item_id"]
#                             if SHOW_TIMING_MATH:
#                                 print(f"Setting start timestamp for new response: {response_start_timestamp_twilio}ms")

#                         await send_mark(websocket, stream_sid)

#                     # Trigger an interruption. Your use case might work better using `input_audio_buffer.speech_stopped`, or combining the two.
#                     if response.get('type') == 'input_audio_buffer.speech_started':
#                         print("Speech started detected.")
#                         if last_assistant_item:
#                             print(f"Interrupting response with id: {last_assistant_item}")
#                             await handle_speech_started_event()
#             except Exception as e:
#                 print(f"Error in send_to_twilio: {e}")

#         async def handle_speech_started_event():
#             """Handle interruption when the caller's speech starts."""
#             nonlocal response_start_timestamp_twilio, last_assistant_item
#             print("Handling speech started event.")
#             if mark_queue and response_start_timestamp_twilio is not None:
#                 elapsed_time = latest_media_timestamp - response_start_timestamp_twilio
#                 if SHOW_TIMING_MATH:
#                     print(f"Calculating elapsed time for truncation: {latest_media_timestamp} - {response_start_timestamp_twilio} = {elapsed_time}ms")

#                 if last_assistant_item:
#                     if SHOW_TIMING_MATH:
#                         print(f"Truncating item with ID: {last_assistant_item}, Truncated at: {elapsed_time}ms")

#                     truncate_event = {
#                         "type": "conversation.item.truncate",
#                         "item_id": last_assistant_item,
#                         "content_index": 0,
#                         "audio_end_ms": elapsed_time
#                     }
#                     await openai_ws.send(json.dumps(truncate_event))

#                 await websocket.send_json({
#                     "event": "clear",
#                     "streamSid": stream_sid
#                 })

#                 mark_queue.clear()
#                 last_assistant_item = None
#                 response_start_timestamp_twilio = None

#         async def send_mark(connection, stream_sid):
#             if stream_sid:
#                 mark_event = {
#                     "event": "mark",
#                     "streamSid": stream_sid,
#                     "mark": {"name": "responsePart"}
#                 }
#                 await connection.send_json(mark_event)
#                 mark_queue.append('responsePart')

#         await asyncio.gather(receive_from_twilio(), send_to_twilio())

async def handle_media_stream_from_file(websocket: WebSocket):
    global buffer_pcm, speech_buffer, stream_sid
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
            while len(buffer_pcm) >= frame_bytes:
                frame = buffer_pcm[:frame_bytes]
                buffer_pcm = buffer_pcm[frame_bytes:]

                is_speech = vad.is_speech(frame, sample_rate)
                if is_speech:
                    speech_buffer += frame
                else:
                   if len(speech_buffer) > 0:
                        # Khi phát hiện silence → xử lý đoạn speech
                        llm_response = await transcribe_and_respond(speech_buffer)
                        speech_buffer = b""
                        if llm_response :
                            """
                            Instead of reading an existing file, we dynamically create one
                            with edge-tts, convert it, and stream it back to Twilio.
                            """
                            print("Start create file")
                            # text_to_speak = "Hello! This is a live text-to-speech test using Edge TTS and Twilio."
                            raw_file = "edge_temp.wav"
                            await generate_tts_wav(llm_response, raw_file)
                            twilio_file = "edge_twilio.wav"
                            convert_to_twilio_format(raw_file, twilio_file)
                            # file_path = "openai_output.wav"
                            file_path = twilio_file
                            with open(file_path, "rb") as f:
                                audio_data = f.read()
                                
                            chunk_size = 160
                            # stream_sid = None

                            try:
                                # async for message in websocket.iter_text():
                                #     data = json.loads(message)

                                #     if data['event'] == 'start':
                                #         stream_sid = data['start']['streamSid']
                                #         print(f"Incoming stream has started {stream_sid}")
                                print("Start send ...")
                                # Sau khi Twilio báo "start", gửi file về
                                for i in range(0, len(audio_data), chunk_size):
                                    chunk = audio_data[i:i+chunk_size]
                                    audio_payload = base64.b64encode(chunk).decode('utf-8')
                        
                                    audio_delta = {
                                        "event": "media",
                                        "streamSid": stream_sid,
                                        "media": {"payload": audio_payload}
                                    }
                                    await websocket.send_json(audio_delta)
                                
                                # Kết thúc stream
                                await websocket.send_json({
                                    "event": "stop",
                                    "streamSid": stream_sid
                                })
                            except Exception as e:
                                print(f"Error: {e}")

# async def handle_media_stream_from_file(websocket: WebSocket):
#     global buffer_pcm, speech_buffer, stream_sid, current_websocket
#     current_websocket = websocket
#     print("✅ Client connected")

#     await websocket.accept()

#     async for message in websocket.iter_text():
#         try:
#             data = json.loads(message)
#         except Exception as e:
#             print("❌ JSON parse error:", e)
#             continue

#         event = data.get("event")

#         if event == "start":
#             stream_sid = data["start"]["streamSid"]
#             print(f"Incoming stream started: {stream_sid}")

#         elif event == "media":
#             # decode μ-law -> PCM16
#             payload_b64 = data["media"]["payload"]
#             ulaw_bytes = base64.b64decode(payload_b64)
#             pcm16_bytes = audioop.ulaw2lin(ulaw_bytes, 2)
#             buffer_pcm += pcm16_bytes

#             # chia frame 30ms
#             while len(buffer_pcm) >= frame_bytes:
#                 frame = buffer_pcm[:frame_bytes]
#                 buffer_pcm = buffer_pcm[frame_bytes:]

#                 is_speech = vad.is_speech(frame, sample_rate)

#                 if is_speech:
#                     speech_buffer += frame
#                 else:
#                     if len(speech_buffer) > 0:
#                         # Khi phát hiện silence → xử lý đoạn speech
#                         llm_response = await transcribe_and_respond(speech_buffer)
#                         speech_buffer = b""

#                         if llm_response:
#                             # convert response -> TTS
#                             raw_file = "edge_temp.wav"
#                             print("llm response:", llm_response)
#                             await generate_tts_wav(llm_response, raw_file)
#                             twilio_file = "edge_twilio.wav"
#                             convert_to_twilio_format(raw_file, twilio_file)
#                             # file_path = "openai_output.wav"
#                             file_path = twilio_file
#                             with open(file_path, "rb") as f:
#                                 audio_data = f.read()
                                
#                             chunk_size = 160
#                             stream_sid = None
#                             print("Sending TTS audio back to Twilio...")
#                             try:
#                                 async for message in websocket.iter_text():
#                                     data = json.loads(message)

#                                     if data['event'] == 'start':
#                                         stream_sid = data['start']['streamSid']
#                                         print(f"Incoming stream has started {stream_sid}")

#                                         # Sau khi Twilio báo "start", gửi file về
#                                         for i in range(0, len(audio_data), chunk_size):
#                                             chunk = audio_data[i:i+chunk_size]
#                                             audio_payload = base64.b64encode(chunk).decode('utf-8')

#                                             audio_delta = {
#                                                 "event": "media",
#                                                 "streamSid": stream_sid,
#                                                 "media": {"payload": audio_payload}
#                                             }
#                                             await websocket.send_json(audio_delta)

#                                         # Kết thúc stream
#                                         await websocket.send_json({
#                                             "event": "stop",
#                                             "streamSid": stream_sid
#                                         })
#                             except Exception as e:
#                                 print(f"Error: {e}")


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