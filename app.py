# ==============================================================================
# PROJECT 2: COMPUTER VISION IMAGE CLASSIFIER (PRODUCTION BUILD)
# ==============================================================================

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# --- 1. PAGE CONFIGURATION ---
# Setting up a professional, wide-layout web interface
st.set_page_config(page_title="Deep Learning Vision Assistant", layout="wide") 
st.title("🧠 Project 2: Computer Vision Image Classifier")
st.write("An enterprise-grade Deep Learning portal to recognize fashion items instantly.")

# --- 2. CACHED DEEP LEARNING PIPELINE ---
# Caching ensures the model trains ONCE and stays in memory for instant client use
@st.cache_resource
def load_and_train_model():
    # Load the official Fashion MNIST dataset
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # Normalize pixel values to scale them between 0.0 and 1.0 for optimal gradient descent
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    # Define the Artificial Neural Network (ANN) Architecture
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),          # Flattens 2D images to 1D vectors (784 pixels)
        layers.Dense(128, activation='relu'),          # Hidden layer with 128 neurons
        layers.Dropout(0.2),                           # Dropout layer to prevent overfitting
        layers.Dense(10, activation='softmax')         # Output layer generating probabilities for 10 classes
    ])
    
    # Compile the model with industry-standard parameters
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train the model silently behind the scenes (verbose=0 hides logs from the client)
    history = model.fit(X_train, y_train, epochs=5, validation_split=0.1, verbose=0)
    
    return model, X_test, y_test

# --- 3. BACKGROUND EXECUTION ---
# Show a clean loading spinner while the neural network initializes
with st.spinner("Initializing Deep Learning Engine... Please wait..."):
    model, X_test, y_test = load_and_train_model()

# Class mapping corresponding to the Fashion MNIST dataset labels
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# --- 4. CLEAN CLIENT INTERFACE ---
st.write("---")  # Visual divider separating initialization from the application
st.subheader("📸 Live AI Prediction Portal")
st.write("Upload a clear photo of any fashion garment or footwear to run an instant AI inference.")

# Split the layout into two clean columns for an optimal user experience
col_upload, col_result = st.columns(2)

with col_upload:
    st.markdown("### 📥 Step 1: Upload Image")
    # File uploader widget built for the client
    uploaded_file = st.file_uploader("Drop your image file here or click to browse", type=["jpg", "jpeg", "png"])

with col_result:
    st.markdown("### 🔮 Step 2: AI Analysis")
    
    # Trigger the prediction pipeline only when a file is provided
    if uploaded_file is not None:
        # 1. Read the uploaded file into raw bytes
        img_bytes = uploaded_file.read()
        
        # 2. Decode image to Grayscale (1 channel) as expected by our network
        raw_img = tf.image.decode_image(img_bytes, channels=1)
        
        # 3. Structural Preprocessing: Resize image to exactly 28x28 pixels
        resized_img = tf.image.resize(raw_img, [28, 28])
        
        # 4. Input Scaling: Normalize pixel values between 0.0 and 1.0
        normalized_img = resized_img / 255.0
        
        # 5. Batch Dimension Injection: Reshaping from (28, 28) to (1, 28, 28)
        final_img = np.expand_dims(normalized_img[:, :, 0], axis=0)
        
        # 6. Execute Model Inference
        predictions = model.predict(final_img, verbose=0)
        predicted_label = np.argmax(predictions)  # Index of highest probability
        confidence = np.max(predictions) * 100    # Confidence percentage
        
        # 7. Display structured results to the client
        st.image(uploaded_file, caption="Target Image", width=220)
        st.metric(label="Detected Item Classification", value=class_names[predicted_label])
        st.progress(int(confidence))
        st.write(f"📊 **AI Confidence Score:** {confidence:.2f}%")
        
    else:
        # Clean placeholders when the application is idle
        st.info("💡 System Ready. Awaiting an image upload from the left panel to execute prediction.")