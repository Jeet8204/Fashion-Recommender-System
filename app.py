
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy.linalg import norm
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential
import os
import pickle
from tqdm import tqdm

# -------------------------------
# 1️⃣ Build the feature extraction model
# -------------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# -------------------------------
# 2️⃣ Get list of image files
# -------------------------------
image_folder = 'archive/images'
file_names = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
print(f"Found {len(file_names)} images.")

# -------------------------------
# 3️⃣ Batch feature extraction
# -------------------------------
def extract_features_batch(image_paths, model, batch_size=32):
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            img = image.load_img(path, target_size=(224,224))
            img_arr = image.img_to_array(img)
            batch_images.append(img_arr)
        batch_images = np.array(batch_images)
        batch_images = preprocess_input(batch_images)
        batch_features = model.predict(batch_images)
        # Flatten and normalize
        batch_features = batch_features.reshape(batch_features.shape[0], -1)
        batch_features = batch_features / np.linalg.norm(batch_features, axis=1, keepdims=True)
        features.extend(batch_features)
    return features

# -------------------------------
# 4️⃣ Extract and save features
# -------------------------------
feature_list = extract_features_batch(file_names, model, batch_size=32)

# Save embeddings and file names
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)

with open('file_names.pkl', 'wb') as f:
    pickle.dump(file_names, f)

print("Feature extraction completed and saved to embeddings.pkl and file_names.pkl")

import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy.linalg import norm
from keras.preprocessing import image
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.models import Sequential
import os
import pickle
from tqdm import tqdm

# -------------------------------
# 1️⃣ Build the feature extraction model
# -------------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalMaxPooling2D()
])

# -------------------------------
# 2️⃣ Get list of image files
# -------------------------------
image_folder = 'archive/images'
file_names = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
print(f"Found {len(file_names)} images.")

# -------------------------------
# 3️⃣ Batch feature extraction
# -------------------------------
def extract_features_batch(image_paths, model, batch_size=32):
    features = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Extracting features"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        for path in batch_paths:
            img = image.load_img(path, target_size=(224,224))
            img_arr = image.img_to_array(img)
            batch_images.append(img_arr)
        batch_images = np.array(batch_images)
        batch_images = preprocess_input(batch_images)
        batch_features = model.predict(batch_images)
        # Flatten and normalize
        batch_features = batch_features.reshape(batch_features.shape[0], -1)
        batch_features = batch_features / np.linalg.norm(batch_features, axis=1, keepdims=True)
        features.extend(batch_features)
    return features

# -------------------------------
# 4️⃣ Extract and save features
# -------------------------------
feature_list = extract_features_batch(file_names, model, batch_size=32)

# Save embeddings and file names
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(feature_list, f)

with open('file_names.pkl', 'wb') as f:
    pickle.dump(file_names, f)

print("Feature extraction completed and saved to embeddings.pkl and file_names.pkl")

