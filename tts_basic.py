import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = "Hola amiga, te quiero contar algo muy bueno! estoy en la casa del bepi y cocinamos altas pastas con una salsa de tomates cherry"
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

