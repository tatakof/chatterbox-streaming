import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = "Hola amiga, te quiero contar algo muy bueno! estoy en la casa del bepi y cocinamos altas pastas con una salsa de tomates cherry"


# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "audio_data/000000_Para la caída del cabello tengo un nuevo champú.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)