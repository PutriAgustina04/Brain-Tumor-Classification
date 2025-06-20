# 🧠 Brain Tumor Classification using CNN on MRI Images

![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-red)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

This application uses Convolutional Neural Networks (CNN) to classify brain tumors from MRI images into 4 categories: **Glioma, Meningioma, Pituitary, and No Tumor**. Built with Streamlit, it enables healthcare professionals and researchers to analyze MRI images efficiently and accurately.

---

## ✨ Features

✅ Upload MRI brain images and receive tumor classification  
✅ Select CNN architecture: VGG16, MobileNetV2, or Xception  
✅ Select learning rate: 0.0001 or 0.00001  
✅ View evaluation metrics: accuracy plot, loss plot, confusion matrix, classification report, F1-score, recall, precision

---

## 🛠 Technologies

- Python  
- TensorFlow / Keras  
- Streamlit  
- Matplotlib  
- Seaborn  
- NumPy  
- Pandas  
- Scikit-learn

---

## 🚀 How to Run

```bash
# 1. Clone this repo
git clone https://github.com/username/brain-tumor-classification.git
cd brain-tumor-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
