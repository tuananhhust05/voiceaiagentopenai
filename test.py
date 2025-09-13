import time
from TTS.api import TTS

# 1. Load model YourTTS
tts = TTS("tts_models/multilingual/multi-dataset/your_tts")

# 2. Text input
text = "Hello, this is a test of the YourTTS model with a custom speaker."


speaker_embedding = tts.synthesizer.tts_model.speaker_manager.compute_embedding_from_clip("customvoice.wav")
print(f"✅ Đã tính toán embedding cho giọng nói từ andrea.wav", speaker_embedding)
# 3. Đo thời gian xử lý
start = time.perf_counter()

# tts.tts_to_file(
#     text=text,
#     # speaker_wav="andrea.wav",
#     speaker_embeddings=speaker_embedding,
#     language="en",
#     file_path="output.wav"
# )

name = "andrea"
tts.synthesizer.tts_model.speaker_manager.name_to_id[name] = 0
tts.synthesizer.tts_model.speaker_manager.embeddings_by_names[name] = [speaker_embedding] 

# 4. Gọi synthesize
start = time.perf_counter()
wav = tts.synthesizer.tts(
    text="Hello with preloaded embedding",
    speaker_name=name,
    language_name="en"
)
tts.synthesizer.save_wav(wav, "output.wav")

end = time.perf_counter()
elapsed = end - start

print(f"✅ Đã tạo file output.wav với giọng Andrea")
print(f"⏱️ Thời gian xử lý: {elapsed:.2f} giây")
