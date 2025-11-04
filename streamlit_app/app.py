# app.py

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import joblib
import gdown
import requests
from io import BytesIO
from tensorflow.keras.models import load_model

# --- CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Dashboard POC – Projet 7", layout="wide")



def load_image_from_drive(file_id):
    """
    Télécharge une image depuis Google Drive à partir de son ID.
    Retourne un objet PIL.Image ou None si échec.
    """
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


# --- CHEMINS LOCAUX ET IDS DRIVE
MODEL_PATH = "model/best_model_baseline_cnn.keras"
CSV_PATH = "data/df_full.csv"
HISTORY_PATH = "outputs/history_baseline_cnn.joblib"
SAMPLE_CSV_PATH = "data/df_sample.csv"

MODEL_DRIVE_ID = "1j69jqMBryuYz0Rk0DC2oc80Cg-LA5inR"
CSV_DRIVE_ID = "1k2bs1DFJ9oawf8twKkY49AfRmmmjBSbu"
HISTORY_DRIVE_ID = "1rA-PNTRfMSX5QP1UtoO3tpVWRKBwA9AC"
SAMPLE_CSV_DRIVE_ID = "1GJbXKgyZTK_B68zlmrMU0SNop4LcR9yo"

# --- TELECHARGEMENT DES FICHIERS SI ABSENTS
download_from_drive(MODEL_PATH, MODEL_DRIVE_ID)
download_from_drive(CSV_PATH, CSV_DRIVE_ID)
download_from_drive(HISTORY_PATH, HISTORY_DRIVE_ID)
# download_from_drive(SAMPLE_CSV_PATH, SAMPLE_CSV_DRIVE_ID)

# --- CHARGEMENT DU MODELE
@st.cache_resource
def load_model_cnn():
    return load_model(MODEL_PATH)

model = load_model_cnn()

# --- CHARGEMENT DES DATAFRAMES
@st.cache_data
def load_dataframe():
    return pd.read_csv(CSV_PATH)

@st.cache_data
def load_sample_dataframe():
    url = f"https://drive.google.com/uc?id={SAMPLE_CSV_DRIVE_ID}"
    return pd.read_csv(url)

# --- TELECHARGEMENT DEPUIS URL DRIVE
import requests
from io import BytesIO

def load_image_from_drive(file_id):
    url = f"https://drive.google.com/uc?id={file_id.strip()}"
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    else:
        return None


df = load_dataframe()
df_sample = load_sample_dataframe()
classes = sorted(df["class"].unique())

# --- PRETRAITEMENT IMAGE
def preprocess_image(img, target_size=(227, 227)):
    img = img.resize(target_size).convert("RGB")
    img_array = np.array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch

# --- INTERFACE PRINCIPALE
st.title("🧪 Dashboard – Preuve de Concept CNN")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Analyse exploratoire",
    "🖼️ Prédiction d'image",
    "📈 Courbes d'entraînement",
    "🧠 Comparaison des modèles"
])

# --- TAB 1 : ANALYSE EXPLORATOIRE
with tab1:
    st.header("📊 Analyse du dataset")
    st.dataframe(df.head())

    class_counts = df["class"].value_counts()
    st.bar_chart(class_counts)

    st.subheader("Exemples d'images par classe")
    selected_class = st.selectbox("Choisir une classe", df_sample["class"].unique())
    sample_ids = df_sample[df_sample["class"] == selected_class]["image_id"].sample(3)

    for file_id in sample_ids:
        img = load_image_from_drive(file_id)
        if img:
            st.image(img, caption=selected_class, use_column_width=True)
        else:
            st.warning(f"Image introuvable pour l'ID : {file_id}")


    


# --- TAB 2 : PREDICTION D'IMAGE
with tab2:
    st.header("🖼️ Prédiction d'une image")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Image chargée", use_column_width=True)

        img_batch = preprocess_image(img)
        y_pred = model.predict(img_batch)
        pred_class = classes[np.argmax(y_pred)]
        st.success(f"Classe prédite : **{pred_class}**")

        probas = pd.Series(y_pred[0], index=classes).sort_values(ascending=False)
        st.bar_chart(probas)

# --- TAB 3 : COURBES D'ENTRAINEMENT
with tab3:
    st.header("📈 Courbes d'entraînement")
    try:
        history = joblib.load(HISTORY_PATH)
        df_hist = pd.DataFrame(history)
        st.line_chart(df_hist[["accuracy", "val_accuracy"]])
        st.line_chart(df_hist[["loss", "val_loss"]])
    except:
        st.warning("Historique non disponible. Vérifiez le fichier .joblib.")

# --- TAB 4 : COMPARAISON DES MODELES
with tab4:
    st.header("🧠 Comparaison CNN vs ICN-T")
    st.info("Le modèle ICN-T sera intégré ici dès qu'il sera entraîné.")
