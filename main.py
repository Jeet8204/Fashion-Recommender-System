import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

# -----------------------
# Page setup
# -----------------------
st.set_page_config(
    page_title="Fashion Gallery",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.title("Fashion Gallery")
st.markdown("<h3>Discover inspiring outfits and find visually similar styles.</h3>", unsafe_allow_html=True)

# -----------------------
# Ensure upload folder
# -----------------------
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -----------------------
# Load feature embeddings and file names
# -----------------------
@st.cache_data
def load_features():
    try:
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        file_names = np.array(pickle.load(open('file_names.pkl', 'rb')))
        # Prepend folder path if needed
        file_names = [os.path.join("images", os.path.basename(f)) for f in file_names]
        return feature_list, file_names
    except FileNotFoundError:
        st.error("Missing 'embeddings.pkl' or 'file_names.pkl' in repo.")
        return None, None

feature_list, file_names = load_features()

# -----------------------
# Load ResNet50 model
# -----------------------
@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
    base_model.trainable = False
    model = Sequential([base_model, GlobalMaxPooling2D()])
    return model

model = load_model()

# -----------------------
# Save uploaded file
# -----------------------
def save_uploaded_file(uploaded_file):
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# -----------------------
# Feature extraction
# -----------------------
def feature_extraction(img_path, model):
    img = Image.open(img_path).convert("RGB").resize((224,224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature = model.predict(img_array, verbose=0).flatten()
    feature = feature / norm(feature)
    return feature

# -----------------------
# Recommend function
# -----------------------
def recommend(feature, feature_list, top_k=5):
    nbrs = NearestNeighbors(n_neighbors=top_k+1, algorithm='brute', metric='euclidean')
    nbrs.fit(feature_list)
    distances, indices = nbrs.kneighbors([feature])
    return indices[0][1:]  # skip the uploaded image itself

# -----------------------
# File uploader UI
# -----------------------
st.markdown("## Upload Your Fashion Item")
uploaded_file = st.file_uploader("Choose an image", type=["jpg","jpeg","png"])

if uploaded_file is not None and feature_list is not None:
    file_path = save_uploaded_file(uploaded_file)
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.subheader("Your Uploaded Item")
        uploaded_img = Image.open(file_path)
        st.image(uploaded_img, width=300)

    with col2:
        features = feature_extraction(file_path, model)
        indices = recommend(features, feature_list, top_k=5)
        st.subheader("Recommended Similar Styles")

        rec_cols = st.columns(5)
        for i, idx in enumerate(indices):
            with rec_cols[i]:
                try:
                    rec_img_path = file_names[idx]
                    rec_img = Image.open(rec_img_path)
                    rec_img.thumbnail((350, 350))  # keeps resolution
                    st.image(rec_img)
                    st.caption(os.path.basename(rec_img_path).split('.')[0].replace('_',' ').title())
                except Exception as e:
                    st.error(f"Cannot load image: {rec_img_path}")
                    st.image("https://via.placeholder.com/350x350.png?text=Image+Missing")
else:
    st.info("Upload an image to discover similar fashion items!")
