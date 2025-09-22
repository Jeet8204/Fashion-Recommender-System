
import streamlit as st
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Sequential
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from pathlib import Path

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.markdown('<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

load_css('style.css')

st.set_page_config(
    page_title="Fashion Gallery",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("Fashion Gallery")
st.markdown("<h3>Discover inspiring outfits and find visually similar styles.</h3>", unsafe_allow_html=True)

if not os.path.exists('uploads'):
    os.makedirs('uploads')

@st.cache_data
def load_features():
    try:
        feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
        file_names = np.array(pickle.load(open('file_names.pkl', 'rb')))
        return feature_list, file_names
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'embeddings.pkl' and 'file_names.pkl' are in the same directory.")
        return None, None

feature_list, file_names = load_features()

@st.cache_resource
def load_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    model = Sequential([base_model, GlobalMaxPooling2D()])
    return model

model = load_model()

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('uploads', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def feature_extraction(img_path, model):
    # Load image and resize
    img = Image.open(img_path).resize((224, 224))
    

    if img.mode == 'RGBA':
        # Convert the image to RGB (3 channels)
        img = img.convert('RGB')
    
    img_array = np.array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img, verbose=0).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list, top_k=5):
    neighbors = NearestNeighbors(n_neighbors=top_k + 1, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0][1:]


st.markdown("## Upload Your Style", unsafe_allow_html=True)
st.markdown("Upload an image of a fashion item you love, and we'll help you discover similar pieces from our collection.", unsafe_allow_html=True)

uploaded_file = st.file_uploader(" ", type=["jpg", "png", "jpeg"], help="Max file size 200MB")

if uploaded_file is not None and feature_list is not None:
    with st.spinner('Curating your gallery...'):
        file_path = save_uploaded_file(uploaded_file)
        
        if file_path:
            st.markdown("<hr style='border:1px solid #eee; margin: 3rem 0;'>", unsafe_allow_html=True)
            
            col_uploaded, col_placeholder = st.columns([1, 4])
            
            with col_uploaded:
                st.subheader("Your Featured Item")
                uploaded_img = Image.open(file_path)
                st.image(uploaded_img, use_container_width=True)
                
            with col_placeholder:
                features = feature_extraction(file_path, model)
                indices = recommend(features, feature_list, top_k=5)

                st.markdown("<h2>Explore Similar Styles</h2>", unsafe_allow_html=True)
                
                cols = st.columns(5)
                
                for i, idx in enumerate(indices):
                    with cols[i]:
                        st.markdown("<div class='recommendation-item'>", unsafe_allow_html=True)
                        try:
                            rec_img = Image.open(file_names[idx])
                            rec_img.thumbnail((350, 350))
                            st.image(rec_img, use_container_width=True)
                            st.caption(os.path.basename(file_names[idx]).replace('_', ' ').split('.')[0].title())
                        except Exception as e:
                            st.error(f"Error loading recommended image: {e}")
                            st.image("https://via.placeholder.com/350x350.png?text=Image+Missing", use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
        else:
            st.error("Failed to save uploaded image.")
elif uploaded_file is None:
    st.info("Upload an image to start discovering!")