import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


# ✅ FIXED: Load the merged model directory (not single checkpoint file)
spanish_model = ChatterboxTTS.from_local("./checkpoints_lora/merged_model", device='cuda:2')

text = "Hola amiga, te quiero contar algo muy bueno!"
print(f"🎯 Generating Spanish TTS for: '{text}'")

wav = spanish_model.generate(text)
ta.save("test-spanish-11.wav", wav, spanish_model.sr)

print(f"✅ Spanish audio saved to: test-spanish-11.wav")
print(f"📊 Audio shape: {wav.shape}")
print(f"⏱️ Duration: {wav.shape[-1] / spanish_model.sr:.2f} seconds")


#audio = spanish_model.generate(
    #"Hola, ¿cómo estás? Me llamo Chatterbox.",
    #audio_prompt_path="spanish_reference.wav"
#)



