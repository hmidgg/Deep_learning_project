# 🧠 Computer Vision Image Classifier (Fashion-MNIST)

An enterprise-grade Deep Learning web portal that trains an Artificial Neural Network (ANN) to classify fashion garments and footwear instantly. Built with TensorFlow, Keras, and Streamlit.

---

## 🚀 Key Features
* **Background Neural Network Training:** Automatic dataset management and weight optimization running behind a clean UI loading spinner.
* **Streamlined Client Interface:** A clean, distraction-free environment allowing end-users to drag-and-drop real garment photos.
* **On-the-Fly Image Processing:** Real-time input preprocessing (decoding, grayscale conversion, scaling, and $28 \times 28$ matrix resizing) hidden under the hood.
* **Live Inference Dashboard:** Displays the prediction metric paired with a visual confidence score tracker.

---

## 🛠️ Technical Stack & Architecture

### 1. Neural Network Structure (ANN)
* **Flatten Input Layer:** Transforms the 2D image matrix ($28 \times 28$ pixels) into a 1D vector of 784 structural inputs.
* **Dense Hidden Layer:** 128 neurons activated via Rectified Linear Unit (`ReLU`).
* **Dropout Regulation:** Set at `0.2` to actively mitigate overfitting during training cycles.
* **Dense Output Layer:** 10 neurons utilizing `Softmax` activation to extract clean categorical probability distributions.

### 2. Implementation Specifications
* **Core Framework:** TensorFlow / Keras (Python 3.10)
* **Optimization Engine:** `Adam` Gradient Descent
* **Loss Evaluation:** `Sparse Categorical Crossentropy`
* **Web UI Framework:** Streamlit (Wide-layout design)

---

## 📦 Local Setup Instructions

Ensure your terminal environment has Anaconda and Python activated, then execute:

```bash
# Clone the repository
git clone [https://github.com/hmidgg/Deep_learning_project.git](https://github.com/hmidgg/Deep_learning_project.git)
cd Deep_learning_project

# Launch the interactive client application
python -m streamlit run app.py
