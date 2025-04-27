def text_to_speech(text):
#     try:
#         tts = gTTS(text=text, lang='en')
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
#             tts.save(fp.name)
#         return fp.name
#     except gTTSError:
#         st.warning("Audio won't be displayed because there is no internet connection.")
#         return None