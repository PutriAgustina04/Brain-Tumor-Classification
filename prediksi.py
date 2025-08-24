import numpy as np
from PIL import Image
from keras.utils import img_to_array

class PrediksiTumor:
    def __init__(self, model):
        self.model = model
        # Ambil ukuran input dari model otomatis (misalnya (None, 224, 224, 3) atau (None, 299, 299, 3))
        self.input_size = model.input_shape[1:3]  # Ambil hanya tinggi & lebar
        self.class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

    def preprocess_image(self, image):
        image = image.resize(self.input_size)  # Resize otomatis sesuai model
        image = img_to_array(image) / 255.0     # Normalisasi
        image = np.expand_dims(image, axis=0)   # Tambah dimensi batch
        return image
    
    # Fungsi untuk Prediksi
    def predict(self, uploaded_file):
        image = Image.open(uploaded_file).convert("RGB")
        processed_image = self.preprocess_image(image)
        prediction = self.model.predict(processed_image)
        predicted_class = self.class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        return predicted_class, confidence
