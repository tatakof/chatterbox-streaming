import torchaudio as ta
from chatterbox.tts import ChatterboxTTS


spanish_model = ChatterboxTTS.from_local("./checkpoints_lora/merged_model", device='cuda:2')
text = "Hola amiga, te quiero contar algo muy bueno! estoy en la casa del bepi, bailando chamamé y cocinando altas pastas con una salsa de tomates cherry"
AUDIO_PROMPT_PATH = "audio_data/000000_Para la caída del cabello tengo un nuevo champú.wav"
wav = spanish_model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH, cfg_weight=0.4)
ta.save("test-spanish-2.wav", wav, spanish_model.sr)


#audio = spanish_model.generate(
    #"Hola, ¿cómo estás? Me llamo Chatterbox.",
    #audio_prompt_path="spanish_reference.wav"
#)



