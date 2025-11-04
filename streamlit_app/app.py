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
# CONFIGURATION DE LA PAGE
# ------------------------------
st.set_page_config(page_title="Dashboard POC – Projet 7", layout="wide")

# # ------------------------------
# # CHEMINS LOCAUX ET URLS HF
# # ------------------------------
# MODEL_PATH = "model/initial_best_model_baseline_cnn.keras"
# CSV_PATH = "data/my_df_full.csv"
# SAMPLE_CSV_PATH = "data/my_df_sample.csv"
# HISTORY_PATH = "outputs/history_baseline_cnn.json"

# HF_MODEL_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/initial_best_model_baseline_cnn.keras"
# HF_CSV_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/df_full.csv"
# HF_SAMPLE_CSV_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/df_sample.csv"
# HF_HISTORY_URL = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/history_baseline_cnn.json"

# # ------------------------------
# # FONCTIONS UTILITAIRES
# # ------------------------------
# def download_from_huggingface(file_path, hf_url):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     response = requests.get(hf_url)
#     if response.status_code == 200:
#         with open(file_path, "wb") as f:
#             f.write(response.content)
#     else:
#         st.error(f"Erreur de téléchargement depuis Hugging Face : {response.status_code}")
#         st.stop()

# def load_image_from_drive(file_id):
#     url = f"https://drive.google.com/uc?id={file_id.strip()}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         try:
#             return Image.open(BytesIO(response.content))
#         except Exception as e:
#             st.warning(f"Erreur de lecture de l'image {file_id} : {e}")
#             return None
#     else:
#         st.warning(f"Impossible de télécharger l'image {file_id} (code {response.status_code})")
#         return None

# def preprocess_image(img, target_size=(227, 227)):
#     img = img.resize(target_size).convert("RGB")
#     img_array = np.array(img) / 255.0
#     img_batch = np.expand_dims(img_array, axis=0)
#     return img_batch

# # ------------------------------
# # CHARGEMENT DES FICHIERS
# # ------------------------------
# @st.cache_resource
# def load_model_cnn():
#     try:
#         download_from_huggingface(MODEL_PATH, HF_MODEL_URL)
#         return load_model(MODEL_PATH)
#     except Exception as e:
#         st.error(f"Erreur de chargement du modèle : {e}")
#         st.stop()

# @st.cache_data
# def load_dataframe():
#     return pd.read_csv(HF_CSV_URL)

# @st.cache_data
# def load_sample_dataframe():
#     return pd.read_csv(HF_SAMPLE_CSV_URL)

# @st.cache_data
# def load_training_history():
#     response = requests.get(HF_HISTORY_URL)
#     return pd.read_json(BytesIO(response.content))



# -------------------------
# --- IMPORTS DES MODULES
# -------------------------
from loaders import (
    load_model_cnn,
    load_dataframe,
    load_sample_dataframe,
    load_training_history,
    load_image_from_drive,
    preprocess_image
)
from loaders import (
    MODEL_PATH,
    HF_MODEL_URL,
    HF_CSV_URL,
    HF_SAMPLE_CSV_URL,
    HF_HISTORY_URL
)



# ------------------------------
# INITIALISATION DES DONNÉES
# ------------------------------
model = load_model_cnn()
df = load_dataframe()
df_sample = load_sample_dataframe()
classes = sorted(df["class"].unique())

# ------------------------------
# INTERFACE STREAMLIT
# ------------------------------
st.title("DASHBOARD – BASELINE CNN VS MODELE ICNT LS")
tab1, tab2, tab3, tab4 = st.tabs([
    "ANALYSE EXPLORATOIRE",
    "PREDICTIONS",
    "COURBES ENTRAINEMENT",
    "COMPARAISON MODELES"
])

class_colors = {
    "normal": "#A6CEE3",
    "benign": "#B2DF8A",
    "malignant": "#FB9A99"
}
color_map = {
    "normal": "#5B8FA8",
    "benign": "#A1C181",
    "malignant": "#D95F02"
}

# ------------------------------
# ONGLET 1 : ANALYSE EXPLORATOIRE
# ------------------------------
with tab1:
    st.header("ANALYSE EXPLORATOIRE")
    st.subheader("Répartition des classes")

    class_counts = df["class"].value_counts()
    labels = class_counts.index.tolist()
    sizes = class_counts.values.tolist()
    colors = [class_colors[cls] for cls in labels]

    col1, col2 = st.columns(2)

    df_bar = class_counts.reset_index()
    df_bar.columns = ["class", "count"]
    fig_bar = px.bar(
        df_bar,
        x="class",
        y="count",
        color="class",
        color_discrete_map=class_colors,
        title="Répartition des classes"
    )
    col1.plotly_chart(fig_bar, use_container_width=True)

    fig, ax = plt.subplots()
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'color': "black", 'fontsize': 12}
    )
    ax.axis('equal')
    plt.setp(autotexts, size=12, weight="bold")
    col2.pyplot(fig)

    st.subheader("Exemples d'images par classe")
    show_normal = st.checkbox("🟦 Classe normal", value=True)
    show_benign = st.checkbox("🟩 Classe benign")
    show_malignant = st.checkbox("🟥 Classe malignant")

    checkbox_map = {
        "normal": show_normal,
        "benign": show_benign,
        "malignant": show_malignant
    }

    for selected_class, is_checked in checkbox_map.items():
        if is_checked:
            st.markdown(
                f"<h4 style='color:{color_map[selected_class]};'>Classe : {selected_class}</h4>",
                unsafe_allow_html=True
            )
            sample_ids = df_sample[df_sample["class"] == selected_class]["image_id"].sample(3)
            cols = st.columns(3)
            for i, file_id in enumerate(sample_ids):
                img = load_image_from_drive(file_id)
                if img:
                    img = img.resize((250, 250))
                    cols[i].image(img, caption=selected_class, use_column_width=False)
                else:
                    cols[i].warning(f"Image introuvable : {file_id}")

# ------------------------------
# ONGLET 2 : PREDICTION D'IMAGE
# ------------------------------
with tab2:
    st.header("PREDICTIONS")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        img = img.resize((250, 250))
        col1, col2 = st.columns([1, 2])
        col1.image(img, caption="Image chargée", use_column_width=False)

        img_batch = preprocess_image(img)
        y_pred = model.predict(img_batch)
        pred_class = classes[np.argmax(y_pred)]
        col2.success(f"CLASSE PREDITE : **{pred_class}**")

        probas = pd.Series(y_pred[0], index=classes).sort_values(ascending=False)
        col2.bar_chart(probas)

# ------------------------------
# ONGLET 3 : COURBES D'ENTRAINEMENT
# ------------------------------
with tab3:
    st.header("COURBES ENTRAINEMENT")
    try:
        history = load_training_history()
        df_hist = pd.DataFrame(history)
        st.line_chart(df_hist[["accuracy", "val_accuracy"]])
        st.line_chart(df_hist[["loss", "val_loss"]])
    except Exception as e:
        st.warning(f"Historique non disponible : {e}")

# ------------------------------
# ONGLET 4 : COMPARAISON MODELES
# ------------------------------
with tab4:
    st.header("COMPARAISON MODELES")
    st.info("Le modèle ICN-T sera intégré ici dès qu'il sera entraîné.")

