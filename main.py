import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import base64

model = load_model("F:/mpi/modelupd.h5")
class_dict = np.load("F:/mpi/artifacts/updclass_names.npy")

def predict(image):
    IMG_SIZE = (1, 224, 224, 3)

    img = image.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)

    # Check if prediction probability for the predicted class is above a certain threshold
    threshold = 0.5
    if pred_proba[0][pred] > threshold:
        return pred, class_dict[pred]
    else:
        return None, "Unknown"

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover;
        color:white; font-family:sans-serif; font-size: 20px;
    }}
    .stButton>button {{
        background-color: #90EE90;
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

contnt = '<p style="font-family:sans-serif; color:White; font-size: 20px;">Herbal medicines are preferred in both developing and developed countries as an alternative to  \
         synthetic drugs mainly because of no side effects. Recognition of these plants by human sight will be \
         tedious, time-consuming, and inaccurate.</p>' \
         '<p style="font-family:sans-serif; color:White; font-size: 20px;">Applications of image processing and computer vision \
         techniques for the identification of the medicinal plants are very crucial as many of them are under \
         extinction as per the IUCN records. Hence, the digitization of useful medicinal plants is crucial  \
         for the conservation of biodiversity.</p>'

if __name__ == '__main__':
    add_bg_from_local("C:/Users/BHARATH KUMAR/Downloads/Medicinal Plant classification/medicinal/artifacts/bg.jpg")
    new_title = '<p style="font-family:sans-serif; color:White; font-size: 42px;">Medicinal Leaf Classification</p>'
    st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(contnt, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file", type=['png', 'jpg'])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)

        img = img.resize((300, 300))
        st.image(img)

        if st.button("PREDICT"):
            pred, name = predict(img)

            if pred is not None:
                result = '<p style="font-family:sans-serif; color:white; font-size: 20px;">The given image is <b>' + name.upper() + '</b></p>'
            else:
                result = '<p style="font-family:sans-serif; color:white; font-size: 20px;">The given image does not belong to any known class.</p>'

            st.markdown(result, unsafe_allow_html=True)
#  "C:\Users\BHARATH KUMAR\Downloads\Medicinal Plant classification\medicinal\Model\model.h5"
#  C:/Users/BHARATH KUMAR/Downloads/Medicinal Plant classification/medicinal/Model/MED2.h5*/