# --------------
# --- IMPORTS
# --------------
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import requests
from io import BytesIO
from tensorflow.keras.models import load_model
import plotly.express as px
import matplotlib.pyplot as plt
# ------------------------------
# CHEMINS LOCAUX ET URLS HF
# ------------------------------
MODEL_PATH = "model/initial_best_model_baseline_cnn.keras"
CSV_PATH = "data/my_df_full.csv"
SAMPLE_CSV_PATH = "data/my_df_sample.csv"
HISTORY_PATH = "outputs/history_baseline_cnn.json"

HF_MODEL_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/initial_best_model_baseline_cnn.keras"
HF_CSV_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/df_full.csv"
HF_SAMPLE_CSV_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/df_sample.csv"
HF_HISTORY_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/history_baseline_cnn.json"

# ------------------------------
# FONCTIONS UTILITAIRES
# ------------------------------
def download_from_huggingface(file_path, hf_url):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    response = requests.get(hf_url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
    else:
        st.error(f"Erreur de téléchargement depuis Hugging Face : {response.status_code}")
        st.stop()

def load_image_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id.strip()}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return Image.open(BytesIO(response.content))
        except Exception as e:
            st.warning(f"Erreur de lecture de l'image {file_id} : {e}")
            return None
    else:
        st.warning(f"Impossible de télécharger l'image {file_id} (code {response.status_code})")
        return None

def preprocess_image(img, target_size=(227, 227)):
    img = img.resize(target_size).convert("RGB")
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# ------------------------------
# CHARGEMENT DES FICHIERS
# ------------------------------
@st.cache_resource
def load_model_cnn():
    try:
        download_from_huggingface(MODEL_PATH, HF_MODEL_URL)
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        st.stop()

@st.cache_data
def load_dataframe():
    return pd.read_csv(HF_CSV_URL)

@st.cache_data
def load_sample_dataframe():
    return pd.read_csv(HF_SAMPLE_CSV_URL)

@st.cache_data
def load_training_history():
    response = requests.get(HF_HISTORY_URL)
    return pd.read_json(BytesIO(response.content))
