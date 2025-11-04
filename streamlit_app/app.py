# ---------------------------
# --- IMPORTS DES LIBRAIRIES
# ---------------------------
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






# -------------------------
# --- IMPORTS DES MODULES
# -------------------------
from loaders import (
    load_model_cnn,
    load_model_ictn,
    load_dataframe,
    load_sample_dataframe,
    load_training_history,
    load_image_from_url,
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

# ------------------------------------------
# COMPOSANTS DE L'INTERFACE IHM STREAMLIT
# ------------------------------------------
# -------------------------------
# --- CONFIGURATION DE LA PAGE
# -------------------------------
st.set_page_config(page_title="Dashboard POC – Projet 7", layout="wide")
st.title("DASHBOARD – BASELINE CNN VS MODELE ICNT LS")

# ---------------------------
# --- DEFINITION DES ONGLETS
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "ANALYSE EXPLORATOIRE",
    "PREDICTIONS",
    "PERFORMANCES",
    "COMPARAISON MODELES"
])
# ----------------------------------
# --- DEFINITION DES CODES COULEURS
# ----------------------------------
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

# ------------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET 1 : ANALYSE EXPLORATOIRE
# ------------------------------------------------------
with tab1:
    st.header("ANALYSE EXPLORATOIRE")
    st.subheader("REPRÉSENTATION EQUILIBRÉE DES CLASSES")

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
        title="DISTRIBUTION DES CLASSES"
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

    st.subheader("APERÇU D'IMAGES PAR CLASSE")
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
            sample_urls = df_sample[df_sample["class"] == selected_class]["image_url"].sample(3)
            cols = st.columns(3)
            for i, url in enumerate(sample_urls):
                img = load_image_from_url(url)
                if img:
                    img = img.resize((250, 250))
                    cols[i].image(img, caption=selected_class, use_column_width=False)
                else:
                    cols[i].warning(f"Image introuvable : {url}")

# ----------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET  2 : PREDICTIONS
# ----------------------------------------------------
with tab2:
    st.header("PREDICTIONS")
    uploaded_file = st.file_uploader("CHOISISSEZ UNE IMAGE", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        import altair as alt

        img = Image.open(uploaded_file).convert("RGB")
        img = img.resize((250, 250))
        img_batch = preprocess_image(img)

        # 🔹 COULEURS ACCESSIBLES TEMPÉRÉES
        model_colors = {
            "BASELINE CNN": "#3B82F6",  # Bleu doux
            "ICTN": "#A78BFA"           # Lavande foncée
        }

        # 🔹 LIGNE 1 : IMAGE À GAUCHE, CHECKBOX À DROITE
        row1_col1, row1_col2 = st.columns([1, 2])
        with row1_col1:
            st.subheader("IMAGE CHARGÉE")
            st.image(img, caption="IMAGE CHARGÉE", use_column_width=False)

        with row1_col2:
    st.subheader("CHOISISSEZ LE(S) MODÈLE(S) À UTILISER")
    cb_col1, cb_col2 = st.columns(2)

        with cb_col1:
            st.markdown(
                f"<h5 style='color:{model_colors['BASELINE CNN']}; font-size:18px;'>BASELINE CNN</h5>",
                unsafe_allow_html=True
            )
            use_baseline = st.checkbox("", value=True)
    
        with cb_col2:
            st.markdown(
                f"<h5 style='color:{model_colors['ICTN']}; font-size:18px;'>ICTN</h5>",
                unsafe_allow_html=True
            )
            use_ictn = st.checkbox("")


        # 🔹 LIGNE 2 : PRÉDICTIONS PAR MODÈLE
        row2_col1, row2_col2 = st.columns(2)

        if use_baseline:
            y_pred_base = model.predict(img_batch)
            pred_base = classes[np.argmax(y_pred_base)]
            with row2_col1:
                st.markdown(
                    f"<h4 style='color:{model_colors['BASELINE CNN']};'>PRÉDICTION – BASELINE CNN</h4>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div style='background-color:{model_colors["BASELINE CNN"]}; padding:10px; border-radius:8px;'>
                        <h5 style='color:white; text-align:center;'>CLASSE PRÉDITE : {pred_base.upper()}</h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if use_ictn:
            try:
                ictn_model = load_model_ictn()  # à définir dans loaders.py
                y_pred_ictn = ictn_model.predict(img_batch)
                pred_ictn = classes[np.argmax(y_pred_ictn)]
                with row2_col2:
                    st.markdown(
                        f"<h4 style='color:{model_colors['ICTN']};'>PRÉDICTION – ICTN</h4>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"""
                        <div style='background-color:{model_colors["ICTN"]}; padding:10px; border-radius:8px;'>
                            <h5 style='color:white; text-align:center;'>CLASSE PRÉDITE : {pred_ictn.upper()}</h5>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                with row2_col2:
                    st.warning(f"ERREUR DE CHARGEMENT DU MODÈLE ICTN : {e}")

        # 🔹 LIGNE 3 : PROBABILITÉS PAR MODÈLE
        row3_col1, row3_col2 = st.columns(2)

        if use_baseline:
            probas_base = pd.Series(y_pred_base[0], index=classes).sort_values(ascending=False)
            df_base = probas_base.reset_index()
            df_base.columns = ["Classe", "Probabilité"]
            with row3_col1:
                st.markdown(
                    f"<h4 style='color:{model_colors['BASELINE CNN']};'>PROBABILITÉS – BASELINE CNN</h4>",
                    unsafe_allow_html=True
                )
                chart_base = alt.Chart(df_base).mark_bar(color=model_colors["BASELINE CNN"]).encode(
                    x=alt.X("Classe", title="CLASSE"),
                    y=alt.Y("Probabilité", title="PROBABILITÉ")
                ).properties(height=300)
                st.altair_chart(chart_base, use_container_width=True)

        if use_ictn:
            probas_ictn = pd.Series(y_pred_ictn[0], index=classes).sort_values(ascending=False)
            df_ictn = probas_ictn.reset_index()
            df_ictn.columns = ["Classe", "Probabilité"]
            with row3_col2:
                st.markdown(
                    f"<h4 style='color:{model_colors['ICTN']};'>PROBABILITÉS – ICTN</h4>",
                    unsafe_allow_html=True
                )
                chart_ictn = alt.Chart(df_ictn).mark_bar(color=model_colors["ICTN"]).encode(
                    x=alt.X("Classe", title="CLASSE"),
                    y=alt.Y("Probabilité", title="PROBABILITÉ")
                ).properties(height=300)
                st.altair_chart(chart_ictn, use_container_width=True)







# -------------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET  3 : COURBES D'ENTRAINEMENT
# -------------------------------------------------------
with tab3:
    st.header("COURBES ENTRAINEMENT")
    try:
        history = load_training_history()
        df_hist = pd.DataFrame(history)
        st.line_chart(df_hist[["accuracy", "val_accuracy"]])
        st.line_chart(df_hist[["loss", "val_loss"]])
    except Exception as e:
        st.warning(f"Historique non disponible : {e}")

# ----------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET  4 : COMPARAISON MODELES
# ----------------------------------------------------
with tab4:
    st.header("COMPARAISON MODELES")
    st.info("Le modèle ICN-T sera intégré ici dès qu'il sera entraîné.")

