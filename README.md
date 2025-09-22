# Fashion-Recommender-System


A Fashion recommender system that allows users to find visually similar clothing items by uploading an image. Built using **Streamlit**, **TensorFlow**, and a pre-trained **ResNet50** model, the application serves as a visual search engine for a fashion gallery.

---

## ✨ Features

- **Visual Search**: Upload an image of any fashion item to find similar products.
- **Deep Learning Model**: Utilizes a pre-trained ResNet50 CNN to extract powerful visual features from images.
- **Efficient Similarity Search**: Employs Spotify's Annoy library for fast and scalable approximate nearest neighbor searches on a dataset of over 44,000 fashion items.
- **Intuitive UI**: A clean, responsive, and elegant user interface built with Streamlit and custom CSS, providing a gallery-like experience.
- **Robustness**: Handles various image formats (including PNGs with transparency) and includes error handling for a smooth user experience.

---

## 🚀 Technologies Used

- **Streamlit**: Web application and user interface.
- **TensorFlow/Keras**: Machine learning model development and inference.
- **ResNet50**: Pre-trained CNN used as a feature extractor.
- **NumPy**: Numerical operations and data handling.
- **Pillow (PIL)**: Image processing.
- **scikit-learn**: `NearestNeighbors` algorithm (used in the brute-force version).
- **Python**: Core programming language.

---
