import os
import streamlit as st
import tensorflow as tf
from prediksi import PrediksiTumor
from evaluasi import EvaluasiModel

# ======== Config Awal ========
st.set_page_config(page_title="Klasifikasi Tumor Otak", page_icon="🧠")

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ======== Helper Functions ========
def centered_header(text):
    st.markdown(f"<h1 style='text-align: center; color: #0077b6;'>{text}</h1>", unsafe_allow_html=True)

def centered_paragraph(text):
    st.markdown(f"<p style='text-align: center;'>{text}</p>", unsafe_allow_html=True)

def show_model_info(classifier):
    model_info = (
        f"📌 Model Aktif ➡️ Arsitektur: `{classifier.architecture}` | "
        f"Learning Rate: `{classifier.learning_rate}`"
        f"Batch Size: `{classifier.batch_size}` | "
    )
    st.markdown(f'<div class="model-info-box">{model_info}</div>', unsafe_allow_html=True)


# ======== Brain Tumor Classifier Class ========
class BrainTumorClassifier:
    def __init__(self, architecture, learning_rate, batch_size):
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.model = None
        self.model_filename = None

    def load_model(self):
        model_name = f"{self.architecture}_lr{self.learning_rate}_bs{self.batch_size}.h5"
        model_path = os.path.join("model", model_name)
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            self.model_filename = model_name
            return True
        return False

    def predict(self, uploaded_file):
        if self.model:
            return PrediksiTumor(self.model).predict(uploaded_file)
        return None

    def get_npz_file(self, data_eval_folder="data_evaluasi"):
        if not self.model_filename:
            return None
        model_base = self.model_filename.replace(".h5", "")
        npz_files = [f for f in os.listdir(data_eval_folder) if f.endswith(".npz") and model_base in f]
        return os.path.join(data_eval_folder, npz_files[0]) if npz_files else None


# ======== Main App Class ========
class AppMain:
    def __init__(self):
        self.selected_page = None

    def setup_sidebar(self):
        # Sidebar Navigation
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("🔍 Prediksi"):
                st.session_state.page = "Prediksi"
        with col2:
            if st.button("📊 Evaluasi"):
                st.session_state.page = "Evaluasi"

        if "page" not in st.session_state:
            st.session_state.page = "Prediksi"

        self.selected_page = st.session_state.page

        # Sidebar Model Settings
        st.sidebar.header("Pengaturan Model")
        architecture = st.sidebar.selectbox("Arsitektur:", ["MobileNetV2", "VGG16", "Xception"])
        learning_rate = st.sidebar.selectbox("Learning Rate:", ["0.001", "0.0001"])
        batch_size = st.sidebar.selectbox("Batch Size:", ["32", "64"])

        if st.sidebar.button("🔄 Muat Model"):
            classifier = BrainTumorClassifier(architecture, learning_rate, batch_size)
            if classifier.load_model():
                st.session_state.classifier = classifier
                st.toast(f"✅ Model berhasil dimuat.")
            else:
                st.sidebar.error("❌ Gagal memuat model. Pastikan file model ada di folder `model`.")

    def run(self):
        self.setup_sidebar()
        classifier = st.session_state.get("classifier", None)

        if self.selected_page == "Prediksi":
            self.page_prediksi(classifier)
        elif self.selected_page == "Evaluasi":
            self.page_evaluasi(classifier)

    def page_prediksi(self, classifier):
        centered_header("🔍 Prediksi Tumor Otak")
        centered_paragraph(
            "Aplikasi ini dirancang menggunakan arsitektur Convolutional Neural Network (CNN) untuk memprediksi citra MRI otak dan mengklasifikasikan jenis tumor seperti glioma, meningioma, pituitary tumor, dan kondisi normal (no tumor) serta menampilkan evaluasi performa model klasifikasi yang telah melalui proses pelatihan sebelumnya."
        )

        if classifier is None or not classifier.model:
            st.warning("⚠️ Model belum dimuat. Silakan muat model terlebih dahulu di sidebar.")
            return

        show_model_info(classifier)

        uploaded_file = st.file_uploader("Unggah Citra MRI", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="🖼️ Citra MRI", use_container_width=True)

            if st.button("🔍 Prediksi Citra"):
                with st.spinner("Memprediksi citra MRI..."):
                    predicted_class, confidence = classifier.predict(uploaded_file)
                    if predicted_class:
                        st.success(f"🧠 Prediksi: **{predicted_class}** ({confidence:.2f}% )")

    def page_evaluasi(self, classifier):
        centered_header("📊 Evaluasi Model Klasifikasi Tumor Otak")
        centered_paragraph(
            "Hasil evaluasi ini menampilkan evaluasi kinerja model dalam mengklasifikasikan glioma, meningioma, pituitary tumor, dan normal (no tumor) berdasarkan citra MRI."
        )

        if classifier is None or not classifier.model:
            st.warning("⚠️ Model belum dimuat. Silakan muat model terlebih dahulu di sidebar.")
            return

        show_model_info(classifier)

        npz_file_path = classifier.get_npz_file()
        if npz_file_path:
            st.toast(f"📂 File evaluasi `{os.path.basename(npz_file_path)}` ditemukan.")

            if st.button("🔍 Evaluasi Model"):
                with st.spinner("Menghitung metrik evaluasi..."):
                    evaluator = EvaluasiModel()
                    y_true, y_pred = evaluator.load_from_npz(npz_file_path)

                    if y_true is None or y_pred is None:
                        st.error("❌ Gagal membaca file evaluasi.")
                        return

                    cm, report, metrics_summary = evaluator.evaluate(y_true, y_pred)

                    st.subheader("📈 Grafik Akurasi dan Loss")
                    evaluator.plot_history()

                    st.subheader("🧩 Confusion Matrix")
                    evaluator.plot_confusion_matrix(cm)

                    st.subheader("📋 Classification Report")
                    evaluator.show_classification_report(report)

                    st.subheader("📈 Metrics Summary")
                    evaluator.show_metrics(metrics_summary)
        else:
            st.warning("⚠️ Tidak ditemukan file evaluasi yang sesuai.")


# ======== Run App ========
if __name__ == "__main__":
    AppMain().run()
