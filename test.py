
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import pickle
import os

# -------------------------------
# 1️⃣ Load dataset embeddings
# -------------------------------
dataset_features = np.array(pickle.load(open('embeddings.pkl', 'rb')))
file_names = np.array(pickle.load(open('file_names.pkl', 'rb')))
print(f"Loaded {len(dataset_features)} features from dataset.")

# -------------------------------
# 2️⃣ Load the feature extraction model
# -------------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# -------------------------------
# 3️⃣ Function to extract features for a single image
# -------------------------------
def extract_feature(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    feature = model.predict(img_arr).flatten()
    feature = feature / norm(feature)
    return feature

# -------------------------------
# 4️⃣ Nearest Neighbors search
# -------------------------------
from sklearn.neighbors import NearestNeighbors
def find_similar_images(query_image_path, dataset_features, file_names, model, top_k=5):
    query_feature = extract_feature(query_image_path, model).reshape(1, -1)
    
    neighbours = NearestNeighbors(n_neighbors=top_k+1, algorithm='brute', metric='euclidean')
    neighbours.fit(dataset_features)
    
    distances, indices = neighbours.kneighbors(query_feature)
    
    print(f"\nQuery Image: {query_image_path}\n")
    print("Top similar images:")
    for idx, distance in zip(indices[0], distances[0]):
        print(file_names[idx], "Distance:", distance)

# -------------------------------
# 5️⃣ Example usage
# -------------------------------
if __name__ == "__main__":
    query_image = 'archive/sample/1.png'  # Replace with your query image
    if not os.path.exists(query_image):
        print(f"Error: {query_image} not found!")
    else:
        find_similar_images(query_image, dataset_features, file_names, model, top_k=5)

import tensorflow as tf
from tensorflow import keras
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm
import pickle
import os

# -------------------------------
# 1️⃣ Load dataset embeddings
# -------------------------------
dataset_features = np.array(pickle.load(open('embeddings.pkl', 'rb')))
file_names = np.array(pickle.load(open('file_names.pkl', 'rb')))
print(f"Loaded {len(dataset_features)} features from dataset.")

# -------------------------------
# 2️⃣ Load the feature extraction model
# -------------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# -------------------------------
# 3️⃣ Function to extract features for a single image
# -------------------------------
def extract_feature(img_path, model):
    img = image.load_img(img_path, target_size=(224,224))
    img_arr = image.img_to_array(img)
    img_arr = np.expand_dims(img_arr, axis=0)
    img_arr = preprocess_input(img_arr)
    feature = model.predict(img_arr).flatten()
    feature = feature / norm(feature)
    return feature

# -------------------------------
# 4️⃣ Nearest Neighbors search
# -------------------------------
from sklearn.neighbors import NearestNeighbors
def find_similar_images(query_image_path, dataset_features, file_names, model, top_k=5):
    query_feature = extract_feature(query_image_path, model).reshape(1, -1)
    
    neighbours = NearestNeighbors(n_neighbors=top_k+1, algorithm='brute', metric='euclidean')
    neighbours.fit(dataset_features)
    
    distances, indices = neighbours.kneighbors(query_feature)
    
    print(f"\nQuery Image: {query_image_path}\n")
    print("Top similar images:")
    for idx, distance in zip(indices[0], distances[0]):
        print(file_names[idx], "Distance:", distance)

# -------------------------------
# 5️⃣ Example usage
# -------------------------------
if __name__ == "__main__":
    query_image = 'archive/sample/1.png'  # Replace with your query image
    if not os.path.exists(query_image):
        print(f"Error: {query_image} not found!")
    else:
        find_similar_images(query_image, dataset_features, file_names, model, top_k=5)

