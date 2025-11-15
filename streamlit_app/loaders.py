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
import tempfile

# ------------------------------------------------------- #
# --- CONFIGURATION CENTRALISEE DES RESSOURCES HF -------- #
# ------------------------------------------------------- #
from config import HF_RESOURCES


# ------------------------------
# FONCTIONS UTILITAIRES
# ------------------------------
def download_from_huggingface(file_path, hf_url):
    """Télécharge une ressource HuggingFace si elle n'existe pas déjà en local."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if not os.path.exists(file_path):
        response = requests.get(hf_url)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
        else:
            st.error(f"Erreur de téléchargement depuis Hugging Face : {response.status_code}")
            st.stop()
    return file_path


def load_image_from_url(url):
    """Charge une image depuis une URL."""
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return Image.open(BytesIO(response.content))
        except Exception as e:
            st.warning(f"Erreur lecture image : {e}")
    else:
        st.warning(f"Image inaccessible : {url} (code {response.status_code})")
    return None


# ------------------------
# --- PRETRAITEMENT CNN
# -----------------------
def preprocess_image_cnn(img, target_size=(227, 227)):
    img = img.resize(target_size).convert("RGB")
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


# ------------------------
# --- PRETRAITEMENT ICNT
# -----------------------
from tensorflow.keras.applications.convnext import preprocess_input as preprocess_convnext

def preprocess_image_icnt(img, target_size=(224, 224)):
    img = img.resize(target_size).convert("RGB")
    img_array = np.array(img)
    img_array = preprocess_convnext(img_array)  # ⚡ même prétraitement que pendant l'entraînement
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch
# ------------------------
# --- PRETRAITEMENT IIV3
# -----------------------
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception

def preprocess_image_iiv3(img, target_size=(224, 224)):
    """
    Prétraitement pour le modèle Improved InceptionV3.
    - Resize à la taille attendue (224x224 dans ton entraînement)
    - Conversion en RGB
    - Normalisation avec preprocess_input (pixels dans [-1, 1])
    - Ajout d'une dimension batch
    """
    img = img.resize(target_size).convert("RGB")
    img_array = np.array(img)
    img_array = preprocess_inception(img_array)  # ⚡ normalisation [-1, 1]
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch



# ------------------------------
# CHARGEMENT DES MODELES
# ------------------------------
@st.cache_resource
def load_model_cnn():
    try:
        res = HF_RESOURCES["models"]["baseline_cnn"]
        download_from_huggingface(res["local"], res["url"])
        return load_model(res["local"])
    except Exception as e:
        st.error(f"Erreur de chargement du modèle CNN : {e}")
        st.stop()


@st.cache_resource
def load_model_ictn():
    try:
        res = HF_RESOURCES["models"]["icnt"]
        download_from_huggingface(res["local"], res["url"])
        return load_model(res["local"])
    except Exception as e:
        st.error(f"Erreur de chargement du modèle ICTN : {e}")
        st.stop()


# @st.cache_resource
# def load_model_iiv3():
#     try:
#         res = HF_RESOURCES["models"]["iiv3"]
#         download_from_huggingface(res["local"], res["url"])
#         return load_model(res["local"])
#     except Exception as e:
#         st.error(f"Erreur de chargement du modèle InceptionV3 : {e}")
#         st.stop()


# ------------------------------
# CHARGEMENT DU MODELE IIV3
# ------------------------------
@st.cache_resource
def load_model_iiv3():
    try:
        res = HF_RESOURCES["models"]["iiv3"]
        download_from_huggingface(res["local"], res["url"])
        return load_model(res["local"])
    except Exception as e:
        st.error(f"Erreur de chargement du modèle IIV3 : {e}")
        st.stop()


# ------------------------------
# CHARGEMENT DES DATASETS
# ------------------------------
@st.cache_data
def load_dataframe():
    res = HF_RESOURCES["datasets"]["full"]
    download_from_huggingface(res["local"], res["url"])
    return pd.read_csv(res["local"])


@st.cache_data
def load_sample_dataframe():
    res = HF_RESOURCES["datasets"]["sample"]
    download_from_huggingface(res["local"], res["url"])
    return pd.read_csv(res["local"])

@st.cache_data
def load_dataset_stats():
    res = HF_RESOURCES["datasets"]["stats"]
    download_from_huggingface(res["local"], res["url"])
    return pd.read_csv(res["local"])
# ------------------------------------------------
# --- CHARGEMENT CSV DATAFRAME IMAGES BLIND TEST
# -----------------------------------------------
import pandas as pd
import requests


from io import StringIO
from config import HF_RESOURCES
import streamlit as st

@st.cache_data
def load_blind_test_sample():
    res = HF_RESOURCES["datasets"]["blind_test"]
    try:
        # Télécharger le CSV depuis HuggingFace
        response = requests.get(res["url"])
        response.raise_for_status()  # lève une erreur si le téléchargement échoue
        df = pd.read_csv(StringIO(response.text))
    except Exception as e:
        # Fallback : charger en local si HuggingFace échoue
        st.warning(f"Impossible de charger le CSV depuis HuggingFace ({e}), utilisation du fichier local.")
        df = pd.read_csv(res["local"])
    return df


# ------------------------------
# CHARGEMENT DES HISTORIQUES
# ------------------------------
@st.cache_data
def load_training_history_cnn():
    res = HF_RESOURCES["history"]["baseline_cnn"]
    download_from_huggingface(res["local"], res["url"])
    return pd.read_json(res["local"])


@st.cache_data
def load_training_history_ictn():
    res = HF_RESOURCES["history"]["icnt"]
    download_from_huggingface(res["local"], res["url"])
    return pd.read_json(res["local"])


# @st.cache_data
# def load_training_history_iiv3():
#     res = HF_RESOURCES["history"]["iiv3"]
#     download_from_huggingface(res["local"], res["url"])
#     return pd.read_json(res["local"])

@st.cache_data
def load_training_history_iiv3():
    """Charge l’historique d’entraînement du modèle IIV3."""
    local_path = HF_RESOURCES["history"]["iiv3"]["local"]
    url = HF_RESOURCES["history"]["iiv3"]["url"]

    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        download_file(url, local_path)

    with open(local_path, "r") as f:
        history = json.load(f)
    return history

