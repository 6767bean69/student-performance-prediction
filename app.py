import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
from collections import Counter

# --- 1. RE-DEFINE CUSTOM KNN (MUST be here to load the model) ---
class CustomKNN:
    def __init__(self, k=5):
        self.k = k
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])
    def _predict_single(self, x):
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return Counter(k_nearest_labels).most_common(1)[0][0]

# --- 2. LOAD MODELS (Cached for speed) ---
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.joblib')
    svm = joblib.load('svm_model.joblib')
    knn = joblib.load('knn_model.joblib')
    ann = tf.keras.models.load_model('ann_model.keras')
    return scaler, svm, knn, ann

scaler, svm_model, knn_model, ann_model = load_models()

# --- 3. BUILD INTERFACE ---
st.title("🎓 Student Performance Predictor")
st.write("Enter student details to see if they are likely to Pass or Fail.")

# User Inputs
hours = st.slider("Study Hours (per week)", 10, 39, 26)
attendance = st.slider("Attendance Rate (%)", 50, 99, 76)
past_score = st.number_input("Past Exam Score", 50, 100, 75)

# --- 4. PREDICT BUTTON ---
if st.button("Predict Result"):
    # Prepare data
    new_data = np.array([[hours, attendance, past_score]])
    scaled_data = scaler.transform(new_data)

    # Get Predictions
    pred_svm = svm_model.predict(scaled_data)[0]
    pred_knn = knn_model.predict(scaled_data)[0]
    pred_ann_prob = ann_model.predict(scaled_data)[0][0]
    pred_ann = 1 if pred_ann_prob > 0.5 else 0

    # Display Results
    st.subheader("Results:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("SVM", "Pass" if pred_svm == 1 else "Fail")
    with col2:
        st.metric("KNN", "Pass" if pred_knn == 1 else "Fail")
    with col3:
        st.metric("ANN", "Pass" if pred_ann == 1 else "Fail", f"Conf: {pred_ann_prob:.2f}")

    # Final Verdict logic
    if (pred_svm + pred_knn + pred_ann) >= 2:
        st.success("✅ Final Verdict: PASS")
    else:
        st.error("❌ Final Verdict: FAIL")