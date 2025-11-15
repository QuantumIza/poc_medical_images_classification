# ---------------------------
# --- IMPORTS DES LIBRAIRIES
# ---------------------------
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import plotly.express as px
import requests
import json
from io import BytesIO

# -------------------------
# --- IMPORTS DES MODULES
# -------------------------
from loaders import (
    load_model_cnn,
    load_model_ictn,
    load_model_iiv3,
    load_dataframe,
    load_sample_dataframe,
    load_blind_test_sample,
    load_training_history_cnn,
    load_training_history_ictn,
    load_training_history_iiv3,
    load_image_from_url,
    preprocess_image_cnn,
    preprocess_image_icnt,
    preprocess_image_iiv3,
    load_dataset_stats
)

# Nettoyage du cache Streamlit au démarrage
st.cache_resource.clear()
st.cache_data.clear()

# ------------------------------
# INITIALISATION DES DONNÉES
# ------------------------------
model = load_model_cnn()
df = load_dataframe()
df_sample = load_sample_dataframe()

# -----------------------------------------------------------------------------
# --- CHARGEMENT DU FICHIER CORRESPONDANCE CLASSES ET PRED POUR BASELINE CNN
# ------------------------------------------------------------------------------
json_url_cnn = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/class_labels_cnn.json"
classes_cnn = json.loads(requests.get(json_url_cnn).text)

# -----------------------------------------------------------------------------
# --- CHARGEMENT DU FICHIER CORRESPONDANCE CLASSES ET PRED POUR MODELE ICNT
# ------------------------------------------------------------------------------
json_url_icnt = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/class_labels_icnt.json"
classes_icnt = json.loads(requests.get(json_url_icnt).text)

# -----------------------------------------------------------------------------
# --- CHARGEMENT DU FICHIER CORRESPONDANCE CLASSES ET PRED POUR MODELE IIV3
# ------------------------------------------------------------------------------
json_url_iiv3 = "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/class_labels_iiv3.json"
classes_iiv3 = json.loads(requests.get(json_url_iiv3).text)

# -------------------------------------------------------------- #
# --- GENERATION DES COMPOSANTS DE L'INTERFACE IHM STREAMLIT --- #
# -------------------------------------------------------------- #

# -------------------------------
# --- 1. CONFIGURATION DE LA PAGE
# -------------------------------
st.set_page_config(
    page_title="Dashboard POC – Projet 7",
    layout="wide"
)
st.title("DASHBOARD – BASELINE CNN VS MODELE ICNT LS")

# ---------------------------
# --- 2. DEFINITION DES ONGLETS
# ---------------------------
tab1, tab2, tab5, tab3, tab4 = st.tabs([
    "ANALYSE EXPLORATOIRE",
    "PREDICTIONS (CNN vs ICNT)",
    "PREDICTIONS (CNN vs IIV3)",   # placé juste après ICNT
    "PERFORMANCES",
    "COMPARAISON MODELES"
])


# ----------------------------------
# --- 3. DEFINITION DES CODES COULEURS
# ----------------------------------
class_colors = {
    "normal": "#A6CEE3",     # Bleu clair
    "benign": "#B2DF8A",     # Vert doux
    "malignant": "#FB9A99"   # Rouge rosé
}

color_map = {
    "normal": "#5B8FA8",     # Bleu plus foncé
    "benign": "#A1C181",     # Vert olive
    "malignant": "#D95F02"   # Orange soutenu
}


# ------------------------------------------------------
# 4. COMPOSANT GRAPHIQUE ONGLET 1 : ANALYSE EXPLORATOIRE
# ------------------------------------------------------
with tab1:
    st.header("ANALYSE EXPLORATOIRE")
    st.subheader("REPRÉSENTATION ÉQUILIBRÉE DES CLASSES")
    st.info("""
    ℹ️ Les images du dataset Kaggle sont déjà prétraitées : resize, normalisation, CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Seule une data augmentation a été appliquée sur la classe *normal* afin de corriger sa sous-représentation.
    """)

    # --- CHARGEMENT DU DATAFRAME DE STATS VIA LOADER
    try:
        df_stats = load_dataset_stats()
        st.subheader("Statistiques du dataset")
        st.dataframe(df_stats)

        # --- BARPLOT COMPARATIF
        fig_stats = px.bar(
            df_stats,
            x="Dataset",
            y="Nombre d'images",
            color="Classe",
            text="Proportion (%)",
            color_discrete_map=class_colors,
            barmode="group",
            title="Répartition des images par dataset et par classe"
        )
        fig_stats.update_traces(textposition="outside")
        st.plotly_chart(fig_stats, use_container_width=True)

    except Exception as e:
        st.warning(f"Impossible de charger les stats : {e}")

    # --- DISTRIBUTION DES CLASSES
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
    ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'color': "black", 'fontsize': 12}
    )
    ax.axis('equal')
    col2.pyplot(fig)

    # --- APERÇU D'IMAGES PAR CLASSE
    st.subheader("APERÇU D'IMAGES PAR CLASSE")

    # Création de 3 colonnes pour les checkboxes côte à côte
    col_normal, col_benign, col_malignant = st.columns(3)

    with col_normal:
        st.markdown(f"<span style='color:{class_colors['normal']}; font-weight:bold;'>Classe normal</span>", unsafe_allow_html=True)
        show_normal = st.checkbox("", value=True, key="cb_normal")

    with col_benign:
        st.markdown(f"<span style='color:{class_colors['benign']}; font-weight:bold;'>Classe benign</span>", unsafe_allow_html=True)
        show_benign = st.checkbox("", key="cb_benign")

    with col_malignant:
        st.markdown(f"<span style='color:{class_colors['malignant']}; font-weight:bold;'>Classe malignant</span>", unsafe_allow_html=True)
        show_malignant = st.checkbox("", key="cb_malignant")

    # Mapping des checkboxes
    checkbox_map = {
        "normal": show_normal,
        "benign": show_benign,
        "malignant": show_malignant
    }

    # Affichage des images selon les cases cochées
    for selected_class, is_checked in checkbox_map.items():
        if is_checked:
            st.markdown(
                f"<h4 style='color:{class_colors[selected_class]};'>Classe : {selected_class}</h4>",
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
# COMPOSANT GRAPHIQUE ONGLET 2 : PREDICTIONS
# ----------------------------------------------------
with tab2:
    st.header("PREDICTIONS")

    # Charger l'échantillon blind test
    df_blind = load_blind_test_sample()

    # Sélecteur d'image
    selected_row = st.selectbox(
        "Choisissez une image du blind test",
        df_blind["source_path"].tolist(),
        key="selectbox_cnn_vs_icnt"
    )

    if selected_row:
        import altair as alt

        # --- Charger l'image depuis HuggingFace
        img_url = selected_row
        img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        img = img.resize((250, 250))
        st.image(img, caption="Image sélectionnée", use_column_width=False)

        # --- Prétraitement de l'image
        img_batch_cnn = preprocess_image_cnn(img, target_size=(227, 227))
        img_batch_ictn = preprocess_image_icnt(img, target_size=(224, 224))

        # 🔹 Couleurs pour les modèles
        model_colors = {
            "BASELINE CNN": "#3B82F6",  # Bleu doux
            "ICTN": "#A78BFA"           # Lavande foncée
        }

        # 🔹 Ligne 1 : Image + choix des modèles
        row1_col1, row1_col2 = st.columns([1, 2])

        with row1_col1:
            st.subheader("IMAGE CHARGÉE")
            # st.image(img, caption="IMAGE CHARGÉE", use_column_width=False)

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

        # 🔹 Ligne 2 : Prédictions par modèle
        row2_col1, row2_col2 = st.columns(2)

        if use_baseline:
            y_pred_base = model.predict(img_batch_cnn)
            pred_base = classes_cnn[np.argmax(y_pred_base)]

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
                ictn_model = load_model_ictn()
                y_pred_ictn = ictn_model.predict(img_batch_ictn)
                pred_ictn = classes_icnt[np.argmax(y_pred_ictn)]

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

        # 🔹 Ligne 3 : Probabilités par modèle
        row3_col1, row3_col2 = st.columns(2)

        if use_baseline:
            probas_base = pd.Series(y_pred_base[0], index=classes_cnn).sort_values(ascending=False)
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
            probas_ictn = pd.Series(y_pred_ictn[0], index=classes_icnt).sort_values(ascending=False)
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



# ----------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET 3 : PREDICTIONS CNN vs IIV3
# ----------------------------------------------------
with tab5:
    st.header("PREDICTIONS (CNN vs IIV3)")

    # Charger le CSV blind test
    df_blind = load_blind_test_sample()

    # Sélecteur d'image
    selected_row = st.selectbox(
        "Choisissez une image du blind test",
        df_blind["source_path"].tolist(),
        key="selectbox_cnn_vs_iiv3"
    )

    if selected_row:
        import altair as alt

        # Charger l'image depuis HuggingFace
        img_url = selected_row
        img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        img = img.resize((250, 250))

        # Prétraitement
        img_batch_cnn = preprocess_image_cnn(img, target_size=(227, 227))
        img_batch_iiv3 = preprocess_image_iiv3(img, target_size=(224, 224))

        # 🔹 Couleurs
        model_colors = {
            "BASELINE CNN": "#3B82F6",
            "IIV3": "#F59E0B"
        }

        # Ligne 1 : Image + choix des modèles
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
                use_baseline = st.checkbox("", value=True, key="cb_cnn_iiv3")

            with cb_col2:
                st.markdown(
                    f"<h5 style='color:{model_colors['IIV3']}; font-size:18px;'>IIV3</h5>",
                    unsafe_allow_html=True
                )
                use_iiv3 = st.checkbox("", key="cb_iiv3")

        # Ligne 2 : Prédictions
        row2_col1, row2_col2 = st.columns(2)

        if use_baseline:
            y_pred_base = model.predict(img_batch_cnn)
            pred_base = classes_cnn[np.argmax(y_pred_base)]
            with row2_col1:
                st.markdown(
                    f"<h4 style='color:{model_colors['BASELINE CNN']};'>PRÉDICTION – BASELINE CNN</h4>",
                    unsafe_allow_html=True
                )
                st.success(f"Classe prédite : {pred_base.upper()}")

        if use_iiv3:
            try:
                iiv3_model = load_model_iiv3()
                y_pred_iiv3 = iiv3_model.predict(img_batch_iiv3)
                pred_iiv3 = classes_iiv3[np.argmax(y_pred_iiv3)]
                with row2_col2:
                    st.markdown(
                        f"<h4 style='color:{model_colors['IIV3']};'>PRÉDICTION – IIV3</h4>",
                        unsafe_allow_html=True
                    )
                    st.success(f"Classe prédite : {pred_iiv3.upper()}")
            except Exception as e:
                with row2_col2:
                    st.warning(f"Erreur de chargement du modèle IIV3 : {e}")

        # Ligne 3 : Probabilités
        row3_col1, row3_col2 = st.columns(2)

        if use_baseline:
            with row3_col1:
                st.markdown(
                    f"<h4 style='color:{model_colors['BASELINE CNN']};'>Probabilités – CNN</h4>",
                    unsafe_allow_html=True
                )
                probas_base = pd.Series(y_pred_base[0], index=classes_cnn).sort_values(ascending=False)
                st.bar_chart(probas_base)

        if use_iiv3:
            with row3_col2:
                st.markdown(
                    f"<h4 style='color:{model_colors['IIV3']};'>Probabilités – IIV3</h4>",
                    unsafe_allow_html=True
                )
                probas_iiv3 = pd.Series(y_pred_iiv3[0], index=classes_iiv3).sort_values(ascending=False)
                st.bar_chart(probas_iiv3)



# -------------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET  4 : COURBES D'ENTRAINEMENT
# -------------------------------------------------------
with tab3:
    st.header("COURBES D'ENTRAINEMENT")

    # --- Historique Baseline CNN
    try:
        history_cnn = load_training_history_cnn()
        df_hist_cnn = pd.DataFrame(history_cnn)
        st.subheader("Baseline CNN")
        st.line_chart(df_hist_cnn[["accuracy", "val_accuracy"]])
        st.line_chart(df_hist_cnn[["loss", "val_loss"]])
    except Exception as e:
        st.warning(f"Historique CNN non disponible : {e}")

    # --- Historique ICNT
    try:
        history_icnt = load_training_history_ictn()
        df_hist_icnt = pd.DataFrame(history_icnt)
        st.subheader("ICNT")
        st.line_chart(df_hist_icnt[["accuracy", "val_accuracy"]])
        st.line_chart(df_hist_icnt[["loss", "val_loss"]])
    except Exception as e:
        st.warning(f"Historique ICNT non disponible : {e}")

    # --- Historique InceptionV3 (si disponible)
    try:
        from loaders import load_training_history_iiv3
        history_iiv3 = load_training_history_iiv3()
        df_hist_iiv3 = pd.DataFrame(history_iiv3)
        st.subheader("InceptionV3")
        st.line_chart(df_hist_iiv3[["accuracy", "val_accuracy"]])
        st.line_chart(df_hist_iiv3[["loss", "val_loss"]])
    except Exception as e:
        st.info("Historique InceptionV3 non disponible ou pas encore entraîné.")

# ----------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET  5 : COMPARAISON MODELES
# ----------------------------------------------------
with tab4:
    st.header("COMPARAISON MODELES")
    st.info("""
    Ici, vous pourrez comparer les performances des différents modèles (Baseline CNN, ICNT, InceptionV3).
    Le modèle ICNT et InceptionV3 seront intégrés dès qu'ils auront été entraînés et leurs historiques disponibles.
    """)


