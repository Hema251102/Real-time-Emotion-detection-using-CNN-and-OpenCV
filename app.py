import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

#  Load trained model
model = tf.keras.models.load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("🎭 Emotion Detection App")
st.sidebar.title("Options")

mode = st.sidebar.selectbox("Choose Mode", ["📷 Real-Time Webcam", "🖼 Upload Image"])

#  Real-time webcam detection
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.expand_dims(face, axis=(0, -1))

            prediction = model.predict(face, verbose=0)
            emotion = emotion_labels[np.argmax(prediction)]

            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return img

if mode == " Real-Time Webcam":
    st.write("**Press Start to use your webcam for emotion detection**")
    webrtc_streamer(key="emotion", video_transformer_factory=EmotionTransformer)

#  Image upload option
elif mode == " Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        # Convert to OpenCV format
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(img_cv, 1.3, 5)

        if len(faces) == 0:
            st.error("No face detected in the image.")
        else:
            for (x, y, w, h) in faces:
                face = img_cv[y:y+h, x:x+w]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (48, 48))
                face = face / 255.0
                face = np.expand_dims(face, axis=(0, -1))

                prediction = model.predict(face)
                emotion = emotion_labels[np.argmax(prediction)]

                st.image(image, caption="Uploaded Image", use_column_width=True)
                st.success(f"### Predicted Emotion: **{emotion}**")
                break  # Predict only first face
