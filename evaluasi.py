import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

class EvaluasiModel:
    def __init__(self):
        self.class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
        self.history = None

    def load_from_npz(self, npz_file):
        """Muat data dari file .npz."""
        try:
            data = np.load(npz_file, allow_pickle=True)

            if "class_names" in data:
                self.class_names = list(map(str, data["class_names"]))

            if "history" in data:
                self.history = data["history"].item()

            y_true = data["y_true"]
            y_pred = data["y_pred"]

            return y_true, y_pred

        except Exception as e:
            st.error(f"Gagal load file: {e}")
            return None, None

    def evaluate(self, y_true, y_pred):
        """Menghitung confusion matrix dan metrik evaluasi."""
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        metrics_summary = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        return cm, report, metrics_summary

    def plot_history(self):
        """Plot grafik training dan validasi."""
        if self.history is None:
            st.warning("History training tidak tersedia.")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.history['accuracy'], label='Train Acc')
        axes[0].plot(self.history['val_accuracy'], label='Val Acc')
        axes[0].set_title('Accuracy per Epoch')
        axes[0].legend()

        axes[1].plot(self.history['loss'], label='Train Loss')
        axes[1].plot(self.history['val_loss'], label='Val Loss')
        axes[1].set_title('Loss per Epoch')
        axes[1].legend()

        st.pyplot(fig)

    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        st.pyplot(plt.gcf())

    def show_classification_report(self, report):
        """Tampilkan classification report dalam tabel."""
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format({
            "precision": "{:.2f}",
            "recall": "{:.2f}",
            "f1-score": "{:.2f}",
            "support": "{:.0f}"
        }))

    def show_metrics(self, metrics_summary):
        """Tampilkan ringkasan metrik."""
        st.metric(label="Accuracy", value=f"{metrics_summary['accuracy']*100:.2f}%")
        st.metric(label="Precision", value=f"{metrics_summary['precision']*100:.2f}%")
        st.metric(label="Recall", value=f"{metrics_summary['recall']*100:.2f}%")
        st.metric(label="F1-Score", value=f"{metrics_summary['f1_score']*100:.2f}%")
