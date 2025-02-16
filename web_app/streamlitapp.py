import streamlit as st
import os
import imageio
import text_to_speech as ts
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Setting up the layout to the streamlit app as wide:
st.set_page_config(layout='wide')

# Setting up the sidebar:
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('Info')
    st.info('This app is originally developed from LipNet deep learning model')

st.title('LipNet Full Stack App')
# Generating a list of options or videos:
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('List of videos', options)

# Generating two columns:
col1, col2 = st.columns(2)

if options:
    # Rendering the video:
    with col1:
        st.info('The video below displays the converted video in MP4 format')
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside the app:
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes)

    with col2:
        st.info('This is all the Machine Learning model sees when making a prediction')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        imageio.mimsave('animation.gif', video, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the Machine Learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text:
        st.info('Decoding the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('UTF-8')
        st.text(converted_prediction)
        ts.text_to_audio(converted_prediction)

        # Reconstructing the audio from text using Coqui TTS:
        st.info('This is the reconstructed audio from the text')
        with open('output.wav', 'rb') as audio_file:
            audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/wav')


