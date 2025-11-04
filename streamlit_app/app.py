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
import plotly.express as px


# --- CONFIGURATION DE LA PAGE
st.set_page_config(page_title="Dashboard POC – Projet 7", layout="wide")


# --- TELECHARGEMENT DEPUIS URL DRIVE

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

# def download_from_drive(file_path, file_id):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     url = f"https://drive.google.com/uc?id={file_id}"
#     gdown.download(url, file_path, quiet=False)

# def download_from_drive(file_path, file_id):
#     os.makedirs(os.path.dirname(file_path), exist_ok=True)
#     url = f"https://drive.google.com/uc?export=download&id={file_id}"
#     response = requests.get(url)
#     if response.status_code == 200:
#         with open(file_path, "wb") as f:
#             f.write(response.content)
#     else:
#         st.error(f"Erreur de téléchargement du fichier {file_id} (code {response.status_code})")
#         st.stop()

def download_from_drive(file_path, file_id):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, file_path, quiet=False)



# --- CHEMINS LOCAUX ET IDS DRIVE
MODEL_PATH = "model/my_initial_best_model_baseline_cnn.keras"
CSV_PATH = "data/my_df_full.csv"
HISTORY_PATH = "outputs/history_baseline_cnn.joblib"
SAMPLE_CSV_PATH = "data/my_df_sample.csv"

MODEL_DRIVE_ID = "1Ia-C-c1JneTNz95_cIccDrpNdNn3dZkQ"  # https://drive.google.com/file/d/1j69jqMBryuYz0Rk0DC2oc80Cg-LA5inR/view?usp=sharing
# https://drive.google.com/file/d/1Ia-C-c1JneTNz95_cIccDrpNdNn3dZkQ/view?usp=sharing
CSV_DRIVE_ID = "12ONyFUXjjlF4d3cY0oKHlgmG_YhPGErI"  # https://drive.google.com/file/d/12ONyFUXjjlF4d3cY0oKHlgmG_YhPGErI/view?usp=sharing
HISTORY_DRIVE_ID = "1rA-PNTRfMSX5QP1UtoO3tpVWRKBwA9AC"
SAMPLE_CSV_DRIVE_ID = "1Hcws4ET-4bup9JLdK1_G6jLyUB8PDWbs"  # https://drive.google.com/file/d/1Hcws4ET-4bup9JLdK1_G6jLyUB8PDWbs/view?usp=sharing

# --- TELECHARGEMENT DES FICHIERS SI ABSENTS
# download_from_drive(MODEL_PATH, MODEL_DRIVE_ID)
download_from_drive(CSV_PATH, CSV_DRIVE_ID)
download_from_drive(HISTORY_PATH, HISTORY_DRIVE_ID)
# download_from_drive(SAMPLE_CSV_PATH, SAMPLE_CSV_DRIVE_ID)

# --- CHARGEMENT DU MODELE
@st.cache_resource
def load_model_cnn():
    try:
        download_from_drive(MODEL_PATH, MODEL_DRIVE_ID)
        if not os.path.exists(MODEL_PATH):
            st.error(f"Fichier non trouvé après téléchargement : {MODEL_PATH}")
            st.stop()
        st.success("Fichier modèle trouvé, tentative de chargement...")
        return load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Erreur de chargement du modèle : {e}")
        st.stop()





model = load_model_cnn()




# --- CHARGEMENT DES DATAFRAMES
@st.cache_data
def load_dataframe():
    return pd.read_csv(CSV_PATH)

@st.cache_data
def load_sample_dataframe():
    url = f"https://drive.google.com/uc?id={SAMPLE_CSV_DRIVE_ID}"
    return pd.read_csv(url)





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

# --- PALETTE DE COULEURS ACCESSIBLES PAR CLASSE
class_colors = {
    "normal": "#A6CEE3",
    "benign": "#B2DF8A",
    "malignant": "#FB9A99"
}


# --- TAB 1 : ANALYSE EXPLORATOIRE
with tab1:
    st.header("📊 Analyse du dataset")

    # Répartition des classes : bar chart + camembert côte à côte
    st.subheader("Répartition des classes")
    class_counts = df["class"].value_counts()
    labels = class_counts.index.tolist()
    sizes = class_counts.values.tolist()
    colors = [class_colors[cls] for cls in labels]

    col1, col2 = st.columns(2)

    # Bar chart interactif avec plotly
    import plotly.express as px
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

    # Camembert avec matplotlib
    import matplotlib.pyplot as plt
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

    # Affichage des exemples d’images par classe
    st.subheader("Exemples d'images par classe")

# Encadrés colorés pour chaque checkbox
show_normal = st.checkbox("🟦 Classe normal", value=True)
show_benign = st.checkbox("🟩 Classe benign")
show_malignant = st.checkbox("🟥 Classe malignant")

checkbox_map = {
    "normal": show_normal,
    "benign": show_benign,
    "malignant": show_malignant
}

color_map = {
    "normal": "#5B8FA8",
    "benign": "#A1C181",
    "malignant": "#D95F02"
}

# Affichage des images par classe cochée
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









    


# --- TAB 2 : PREDICTION D'IMAGE
with tab2:
    st.header("🖼️ Prédiction d'une image")
    uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        img = img.resize((250, 250))  # Redimensionne l'image pour un affichage plus compact

        col1, col2 = st.columns([1, 2])

        # Affichage de l'image
        col1.image(img, caption="Image chargée", use_column_width=False)

        # Prédiction
        img_batch = preprocess_image(img)
        y_pred = model.predict(img_batch)
        pred_class = classes[np.argmax(y_pred)]

        col2.success(f"Classe prédite : **{pred_class}**")

        # Affichage des probabilités
        probas = pd.Series(y_pred[0], index=classes).sort_values(ascending=False)
        col2.bar_chart(probas)


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
