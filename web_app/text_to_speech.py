from TTS.api import TTS

# Initialize the TTS model (female voice)
# tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=True).to("cpu")


tts = TTS(model_name='tts_models/en/vctk/vits', progress_bar=True)
# print('Available speakers: ', tts.speakers)


# Convert text to speech and save as a WAV file:
def text_to_audio(text, speaker='p228', file_path='output.wav'):
    tts.tts_to_file(text=text, speaker=speaker, file_path=file_path)



# import simpleaudio as sa
#
# wave_obj = sa.WaveObject.from_wave_file("output.wav")
# play_obj = wave_obj.play()
# play_obj.wait_done()

