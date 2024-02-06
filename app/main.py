import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

#Title
st.title("Brain Tumor classification")

#Header
st.header('Upload your image')

#Upload file
uploaded_file = st.file_uploader("", type=['img', 'png', 'jpeg', 'jpg'])

