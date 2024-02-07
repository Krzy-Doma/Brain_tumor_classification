import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import sys
import path

dir = path.Path(__file__).abspath()
sys.path.append(dir.parent.parent)
path_to_model = '../saved_models/2.h5'


class_names = ['Glioma', 'Meningioma', 'No tumor', 'Pituitary Adenoma']

input_shape = (256, 256, 3)

st.set_page_config(page_title='Brain tumor classfication')

#Title
st.title("Brain Tumor classification")

#Description
st.write(f"Welcome to the brain tumor classification tool powered by Convolutional Neural Networks! This website allows you to upload an MRI image of a brain, which is then analyzed by an advanced CNN model to predict one of four possible classes: **'Glioma', 'Meningioma', 'No tumor', or 'Pituitary Adenoma'**. The model achieved 95% accuracy on the test set. Simply upload your MRI image and let the CNN model offer insights into your brain's health status.")

st.write(f"**Glioma:** A type of brain tumor that starts in the glial cells, which support nerve cells in the brain. They can be either cancerous or non-cancerous.")
st.write(f"**Meningioma:** A usually non-cancerous brain tumor that forms in the meninges, the protective layers surrounding the brain and spinal cord. It tends to grow slowly over time.")
st.write(f"**Pituitary Adenoma:** A benign tumor that develops in the pituitary gland, located at the base of the brain. It can disrupt hormone production and cause various symptoms depending on the affected hormones.")


#Header
st.header('Upload MRI image')


#Model
def load_model():
    model = tf.keras.models.load_model(path_to_model)
    return model

with st.spinner("Loading Model...."):
    model = load_model()
    
    
#Upload file
file = st.file_uploader("", type=['png', 'jpeg', 'jpg'], label_visibility='hidden')
    
    
def preprocess_image(uploaded_file):
    # Open the uploaded image using PIL
    img = Image.open(uploaded_file)

    # Resize image to match required input shape
    img = img.resize((input_shape[1], input_shape[0]))

    # Convert image to numpy array
    img_array = np.array(img)

    # Ensure image is in RGB format
    if img_array.shape[-1] != 3:
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.concatenate([img_array, img_array, img_array], axis=-1)

    # Normalize pixel values to [0, 1]
    img_array = img_array / 255.0

    return img_array
 

predicted_class = None
confidence = None

c1, c2, c3 = st.columns(3)
    
#Display image
if file is not None:
    img = preprocess_image(file)
    img = np.expand_dims(img, axis=0)
    
    #Show image
    c2.image(file, caption='Uploaded Image', use_column_width=True)
     
    prediction = model.predict(img)
    
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = np.max(prediction[0])
    
#Display prediction    
st.write("Press the button below to classify")

button = st.button("Classify", type="primary")
if predicted_class and confidence and button:
    st.subheader(f'Predicted class: {predicted_class}')
    st.subheader(f'Confidence     : {confidence*100}%')
else:
    st.subheader(f'Predicted class: ---------')
    st.subheader(f'Confidence     : ---------')
    

st.divider()

st.write("")    
github = 'https://github.com/Krzy-Doma/brain_tumor_classification'
st.write(f"Check out my [Github](%s)" % github)
