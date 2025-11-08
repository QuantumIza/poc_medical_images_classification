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
import gdown
# ------------------------------
# CHEMINS LOCAUX ET URLS HF
# ------------------------------
MODEL_CNN_PATH = "model/best_model_baseline_cnn.keras"
MODEL_ICNT_PATH = "model/best_model_icnt.keras"

CSV_PATH = "data/my_df_full.csv"
SAMPLE_CSV_PATH = "data/my_df_sample.csv"
HISTORY_PATH = "outputs/history_baseline_cnn.json"

HF_MODEL_CNN_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/best_model_baseline_cnn.keras"
HF_MODEL_ICNT_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/best_model_icnt.keras"
HF_CSV_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/df_full.csv"
HF_SAMPLE_CSV_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/df_sample.csv"
HF_HISTORY_CNN_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/history_baseline_cnn.json"
HF_HISTORY_ICNT_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/history_icnt.json"

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



# def load_image_from_drive(file_id):
#     url = f"https://drive.google.com/uc?id={file_id.strip()}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         try:
#             return Image.open(BytesIO(response.content))
#         except Exception as e:
#             st.warning(f"Erreur de lecture de l'image {file_id} : {e}")
#             return None
# --- stabilite chargement images avec gdown
import gdown
import tempfile

# def load_image_from_drive(file_id):
#     url = f"https://drive.google.com/uc?id={file_id.strip()}"
#     try:
#         with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
#             gdown.download(url, tmp.name, quiet=True)
#             return Image.open(tmp.name)
#     except Exception as e:
#         st.warning(f"Erreur de lecture de l'image {file_id} : {e}")
#         return None

def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return Image.open(BytesIO(response.content))
        except Exception as e:
            st.warning(f"Erreur lecture image : {e}")
    else:
        st.warning(f"Image inaccessible : {url} (code {response.status_code})")
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
        download_from_huggingface(MODEL_CNN_PATH, HF_MODEL_CNN_URL)
        return load_model(MODEL_CNN_PATH)
    except Exception as e:
        st.error(f"Erreur de chargement du modèleCNN : {e}")
        st.stop()
@st.cache_resource
def load_model_ictn():
    try:
        download_from_huggingface(MODEL_ICNT_PATH, HF_MODEL_CNN_URL) # download_from_huggingface("model/ictn_model.keras", "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/ictn_model.keras")
        return load_model(MODEL_ICNT_PATH) # return load_model("model/ictn_model.keras")
    except Exception as e:
        st.error(f"Erreur de chargement du modèle ICTN : {e}")
        st.stop()


@st.cache_data
def load_dataframe():
    return pd.read_csv(HF_CSV_URL)

@st.cache_data
def load_sample_dataframe():
    return pd.read_csv(HF_SAMPLE_CSV_URL)

@st.cache_data
def load_training_history_cnn():
    response = requests.get(HF_HISTORY_CNN_URL)
    return pd.read_json(BytesIO(response.content))

@st.cache_data
def load_training_history_ictn():
    response = requests.get(HF_HISTORY_ICTN_URL)
    return pd.read_json(BytesIO(response.content))
