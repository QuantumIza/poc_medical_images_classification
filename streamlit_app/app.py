import streamlit as st
import gc
st.markdown(
    """
    <style>
    /* Largeur maximale pour le contenu central */
    .main {
        max-width: 1000px;
        margin: auto;
    }

    /* Titres principaux */
    h1, h2, h3 {
        font-size: 26px !important;
        font-weight: 600 !important;
    }

    /* Sous-titres Streamlit */
    .stSubheader {
        font-size: 22px !important;
        font-weight: 500 !important;
    }

    /* Labels des widgets (selectbox, sliders, etc.) */
    div[data-testid="stSelectboxLabel"] > label {
        font-size: 20px !important;
        font-weight: 600 !important;
        color: #222 !important;
    }

    /* Captions */
    div[data-testid="stCaption"] {
        font-size: 18px !important;
        color: #333 !important;
    }

    /* Tableaux */
    .stDataFrame {
        font-size: 15px !important;
    }
    /* Label du widget (ex: "Image du blind test") */
    label[data-testid="stWidgetLabel"] {
        font-size: 20px !important;
        font-weight: 600 !important;
        color: #222 !important;
    }

    /* Options de la selectbox */
    div[data-testid="stMarkdownContainer"] p {
        font-size: 18px !important;
        color: #333 !important;
    }

    /* Liste déroulante (options visibles) */
    div[role="listbox"] div {
        font-size: 20px !important;
        color: #333 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)



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
from config import HF_PERFORMANCES
from config import HF_COMPARAISON 

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
st.title("DASHBOARD – CLASSIFICATION MAMMOGRAPHIES - PREUVE DE CONCEPT")

# ---------------------------
# --- 2. DEFINITION DES ONGLETS
# ---------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "ANALYSE EXPLORATOIRE",
    "PREDICTIONS CNN vs ICNT",
    "PREDICTIONS CNN vs IIV3",
    "PERFORMANCES",
    "COMPARAISON MODELES"
])


# ----------------------------------
# --- 3. DEFINITION DES CODES COULEURS
# ----------------------------------
# class_colors = {
#     "normal": "#A6CEE3",     # Bleu clair
#     "benign": "#B2DF8A",     # Vert doux
#     "malignant": "#FB9A99"   # Rouge rosé
# }
# class_colors = {
#     "normal": "#4E79A7",     # Bleu doux/grisé
#     "benign": "#7B6D5D",     # Taupe/gris-brun élégant
#     "malignant": "#8A70C9"   # Violet lumineux mais pas agressif
# }
class_colors = {
    "normal": "#4E79A7",     # Bleu doux/grisé
    "benign": "#E07B7B",     # Corail rosé feutré
    "malignant": "#8A70C9"   # Violet lumineux mais pas agressif
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
        # --- TEXTE EXPLICATIF
        st.markdown(
            "<p style='font-size:16px; color:black;'>
            📊 Chaque groupe correspond à un sous-ensemble du dataset : 
            <ul>
                <li>IMAGES CHARGEES DEPUIS KAGGLE</li>
                <li>IMAGES APRES REEQUILIBRAGES DES CLASSES (data augmentation ciblee sur la clase "normal"), SUBDIVIS2ES EN ECHANTILLON ENTARINEMENT ET ECHANTILLON BLID TEST</li>
            </ul>
            <b>total</b>, <b>entraînement</b> et <b>blind test</b>. Les couleurs indiquent les classes (<span style='color:#4E79A7;'>normal</span>, <span style='color:#7B6D5D;'>benign</span>, <span style='color:#8A70C9;'>malignant</span>).</p>",
            unsafe_allow_html=True
        )



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

    # # --- APERÇU D'IMAGES PAR CLASSE
    st.subheader("APERÇU D'IMAGES PAR CLASSE")
    # --- BOUCLE SUR LES TROIS CLASSES
    for selected_class in ["normal", "benign", "malignant"]:
        with st.expander(f"Classe : {selected_class.capitalize()}"):
            # --- SELECTION ALEATOIRE DE TROIS IMAGES POUR LA CLASSE
            sample_urls = df_sample[df_sample["class"] == selected_class]["image_url"].sample(3, random_state=42)
            cols = st.columns(3)
            for i,url in enumerate(sample_urls):
                img = load_image_from_url(url)
                if img:
                    img = img.resize((220,220))  # --- TAILLE REDUITE POUR EPARGNER MEMOIRE
                    cols[i].image(img, caption=f"{selected_class.capitalize()} - Exemple {i+1}", use_column_width=False)
                else:
                    cols[i].warning(f"Image introuvable : {url}")

    # # Création de 3 colonnes pour les checkboxes côte à côte
    # col_normal, col_benign, col_malignant = st.columns(3)

    # with col_normal:
    #     # st.markdown(f"<span style='color:{class_colors['normal']}; font-weight:bold;'>Classe normal</span>", unsafe_allow_html=True)
    #     st.subheader("Classe normal")
    #     show_normal = st.checkbox("", value=True, key="cb_normal")

    # with col_benign:
    #     # st.markdown(f"<span style='color:{class_colors['benign']}; font-weight:bold;'>Classe benign</span>", unsafe_allow_html=True)
    #     st.subheader("Classe benign")
    #     show_benign = st.checkbox("", key="cb_benign")

    # with col_malignant:
    #     # st.markdown(f"<span style='color:{class_colors['malignant']}; font-weight:bold;'>Classe malignant</span>", unsafe_allow_html=True)
    #     st.subheader("Classe malignant")
    #     show_malignant = st.checkbox("", key="cb_malignant")

    # # Mapping des checkboxes
    # checkbox_map = {
    #     "normal": show_normal,
    #     "benign": show_benign,
    #     "malignant": show_malignant
    # }

    # # Affichage des images selon les cases cochées
    # for selected_class, is_checked in checkbox_map.items():
    #     if is_checked:
    #         st.markdown(
    #             f"<h4 style='color:{class_colors[selected_class]};'>Classe : {selected_class}</h4>",
    #             unsafe_allow_html=True
    #         )
    #         sample_urls = df_sample[df_sample["class"] == selected_class]["image_url"].sample(3)
    #         cols = st.columns(3)
    #         for i, url in enumerate(sample_urls):
    #             img = load_image_from_url(url)
    #             if img:
    #                 img = img.resize((250, 250))
    #                 cols[i].image(img, caption=selected_class, use_column_width=False)
    #             else:
    #                 cols[i].warning(f"Image introuvable : {url}")
    





# ----------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET 2 : PREDICTIONS CNN vs ICTN
# ----------------------------------------------------
with tab2:
    st.header("COMPARAISON DES PRÉDICTIONS : BASELINE CNN VS CONVNEXT-TINY")
    import psutil
    st.write(f"RAM utilisée : {psutil.virtual_memory().percent}%")

    ictn_model = load_model_ictn()
    # Libérer IIV3 si présent
    if "iiv3_model" in globals():
        del iiv3_model
        gc.collect()

    # --- Palette harmonisée
    model_colors = {
        "BASELINE CNN": "#4E79A7",   # Bleu doux/grisé
        "ICTN": "#5B7C5B"            # Violet lumineux mais feutré
    }

    # --- BLOC 1 : SELECTION IMAGE
    st.markdown(
        """
        <div style="border:2px solid #5A2D82; border-radius:8px;
                    padding:12px; background-color:#F9F6FB; margin:20px 0;">
            <div style="font-size:20px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                Sélectionnez une image du blind test pour comparer les prédictions des deux modèles.
            </div>
        """,
        unsafe_allow_html=True
    )
    df_blind = load_blind_test_sample()
    # --- ON CUSTOMISE LES LIBELLES DE LA LISTE DEROULANTE
    # Créer une colonne "label" plus lisible
    df_blind["label"] = df_blind["source_path"].apply(lambda x: x.split("/")[-1])
    
    # Utiliser cette colonne comme affichage dans la selectbox
    selected_label = st.selectbox(
        "Image du blind test",
        df_blind["label"].tolist(),
        key="selectbox_cnn_vs_icnt"
    )
    
    # Récupérer l’URL correspondant au label choisi
    selected_url = df_blind.loc[df_blind["label"] == selected_label, "source_path"].values[0]

    # ----------------------------------------------------
   

    if selected_url:
        img_url = selected_url
        img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
        img = img.resize((250, 250))
        st.image(img, caption="Image sélectionnée", use_column_width=False)
    st.markdown("</div>", unsafe_allow_html=True)

    if selected_url:
        import altair as alt

        # --- PRETRAITEMENT
        img_batch_cnn = preprocess_image_cnn(img, target_size=(227, 227))
        img_batch_ictn = preprocess_image_icnt(img, target_size=(224, 224))

        # --- BLOC 2 : CHOIX DES MODELES
        st.markdown(
            """
            <div style="border:2px solid #5A2D82; border-radius:8px;
                        padding:12px; background-color:#F9F6FB; margin:20px 0;">
                <div style="font-size:20px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                    ⚙️ Choisissez les modèles à comparer
                </div>
            """,
            unsafe_allow_html=True
        )
        cb_col1, cb_col2 = st.columns(2)
        with cb_col1:
            st.markdown(
                f"<h5 style='color:{model_colors['BASELINE CNN']}; font-size:18px;'>BASELINE CNN</h5>",
                unsafe_allow_html=True
            )
            use_baseline = st.checkbox("PREDIRE AVEC BASELINE CNN", value=True, key="checkbox_baseline")
        with cb_col2:
            st.markdown(
                f"<h5 style='color:{model_colors['ICTN']}; font-size:18px;'>ICTN</h5>",
                unsafe_allow_html=True
            )
            use_ictn = st.checkbox("PREDIRE AVEC ICNT", value=True, key="checkbox_ictn")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BLOC 3 : PREDICTIONS
        row2_col1, row2_col2 = st.columns(2)
        pred_base, pred_ictn = None, None

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
                    <div style="border:2px solid {model_colors['BASELINE CNN']};
                                border-radius:8px; padding:12px; background:#FAFAFA;">
                        <h5 style="color:{model_colors['BASELINE CNN']}; text-align:center; margin:0;">
                            CLASSE PRÉDITE : {pred_base.upper()}
                        </h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )



        
        if use_ictn:
            try:
                # ictn_model = load_model_ictn()
                y_pred_ictn = ictn_model.predict(img_batch_ictn)
                pred_ictn = classes_icnt[np.argmax(y_pred_ictn)]
        
                with row2_col2:
                    st.markdown(
                        f"<h4 style='color:{model_colors['ICTN']};'>PRÉDICTION – ICTN</h4>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"""
                        <div style="border:2px solid {model_colors['ICTN']};
                                    border-radius:8px; padding:12px; background:#FAFAFA;">
                            <h5 style="color:{model_colors['ICTN']}; text-align:center; margin:0;">
                                CLASSE PRÉDITE : {pred_ictn.upper()}
                            </h5>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            except Exception as e:
                with row2_col2:
                    st.warning(f"Erreur de chargement du modèle ICTN : {e}")


        # --- BLOC 4 : PROBABILITES
        row3_col1, row3_col2 = st.columns(2)
        if use_baseline:
            probas_base = pd.Series(y_pred_base[0], index=classes_cnn).sort_values(ascending=False)
            df_base = probas_base.reset_index()
            df_base.columns = ["Classe", "Probabilité"]
            with row3_col1:
                st.markdown(
                    f"<h4 style='color:{model_colors['BASELINE CNN']};'> PROBABILITÉS – BASELINE CNN</h4>",
                    unsafe_allow_html=True
                )
                chart_base = alt.Chart(df_base).mark_bar(color=model_colors["BASELINE CNN"]).encode(
                    x=alt.X("Classe", title="CLASSE"),
                    y=alt.Y("Probabilité", title="PROBABILITÉ")
                ).properties(height=300)
                st.altair_chart(chart_base, use_container_width=True)
            # Libération mémoire après affichage
            if "y_pred_base" in locals():
                del y_pred_base, probas_base, df_base, chart_base
                gc.collect()

        if use_ictn:
            probas_ictn = pd.Series(y_pred_ictn[0], index=classes_icnt).sort_values(ascending=False)
            df_ictn = probas_ictn.reset_index()
            df_ictn.columns = ["Classe", "Probabilité"]
            with row3_col2:
                st.markdown(
                    f"<h4 style='color:{model_colors['ICTN']};'> PROBABILITÉS – ICTN</h4>",
                    unsafe_allow_html=True
                )
                chart_ictn = alt.Chart(df_ictn).mark_bar(color=model_colors["ICTN"]).encode(
                    x=alt.X("Classe", title="CLASSE"),
                    y=alt.Y("Probabilité", title="PROBABILITÉ")
                ).properties(height=300)
                st.altair_chart(chart_ictn, use_container_width=True)
            # ✅ Libération mémoire après affichage
            if "y_pred_ictn" in locals():
                del y_pred_ictn, probas_ictn, df_ictn, chart_ictn
                gc.collect()

       




# ----------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET 3 : PREDICTIONS CNN vs IIV3
# ----------------------------------------------------
# with tab3:
#     st.header("PREDICTIONS BASELINE CNN VS IMPROVED INCEPTIONV3")

#     # Charger le CSV blind test
#     df_blind = load_blind_test_sample()

#     # Sélecteur d'image
#     selected_row_iiv3  = st.selectbox(
#         "Choisissez une image du blind test",
#         df_blind["source_path"].tolist(),
#         key="selectbox_cnn_vs_iiv3"
#     )

#     if selected_row_iiv3 :
#         import altair as alt

#         # Charger l'image depuis HuggingFace
#         img_url = selected_row_iiv3 
#         img = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
#         img = img.resize((250, 250))

#         # Prétraitement
#         img_batch_cnn = preprocess_image_cnn(img, target_size=(227, 227))
#         img_batch_iiv3 = preprocess_image_iiv3(img, target_size=(224, 224))

#         # 🔹 Couleurs
#         model_colors = {
#             "BASELINE CNN": "#3B82F6",
#             "IIV3": "#F59E0B"
#         }

#         # Ligne 1 : Image + choix des modèles
#         row1_col1, row1_col2 = st.columns([1, 2])

#         with row1_col1:
#             st.subheader("IMAGE CHARGÉE")
#             st.image(img, caption="IMAGE CHARGÉE", use_column_width=False)

#         with row1_col2:
#             st.subheader("CHOISISSEZ LE(S) MODÈLE(S) À UTILISER")
#             cb_col1, cb_col2 = st.columns(2)

#             with cb_col1:
#                 st.markdown(
#                     f"<h5 style='color:{model_colors['BASELINE CNN']}; font-size:18px;'>BASELINE CNN</h5>",
#                     unsafe_allow_html=True
#                 )
#                 use_baseline = st.checkbox("", value=True, key="cb_cnn_iiv3")

#             with cb_col2:
#                 st.markdown(
#                     f"<h5 style='color:{model_colors['IIV3']}; font-size:18px;'>IIV3</h5>",
#                     unsafe_allow_html=True
#                 )
#                 use_iiv3 = st.checkbox("", key="cb_iiv3")

#         # Ligne 2 : Prédictions
#         row2_col1, row2_col2 = st.columns(2)

#         if use_baseline:
#             y_pred_base = model.predict(img_batch_cnn)
#             pred_base = classes_cnn[np.argmax(y_pred_base)]
#             with row2_col1:
#                 st.markdown(
#                     f"<h4 style='color:{model_colors['BASELINE CNN']};'>PRÉDICTION – BASELINE CNN</h4>",
#                     unsafe_allow_html=True
#                 )
#                 st.success(f"Classe prédite : {pred_base.upper()}")

#         if use_iiv3:
#             try:
#                 iiv3_model = load_model_iiv3()
#                 y_pred_iiv3 = iiv3_model.predict(img_batch_iiv3)
#                 pred_iiv3 = classes_iiv3[np.argmax(y_pred_iiv3)]
#                 with row2_col2:
#                     st.markdown(
#                         f"<h4 style='color:{model_colors['IIV3']};'>PRÉDICTION – IIV3</h4>",
#                         unsafe_allow_html=True
#                     )
#                     st.success(f"Classe prédite : {pred_iiv3.upper()}")
#             except Exception as e:
#                 with row2_col2:
#                     st.warning(f"Erreur de chargement du modèle IIV3 : {e}")

#         # Ligne 3 : Probabilités
#         row3_col1, row3_col2 = st.columns(2)

#         if use_baseline:
#             with row3_col1:
#                 st.markdown(
#                     f"<h4 style='color:{model_colors['BASELINE CNN']};'>Probabilités – CNN</h4>",
#                     unsafe_allow_html=True
#                 )
#                 probas_base = pd.Series(y_pred_base[0], index=classes_cnn).sort_values(ascending=False)
#                 st.bar_chart(probas_base)

#         if use_iiv3:
#             with row3_col2:
#                 st.markdown(
#                     f"<h4 style='color:{model_colors['IIV3']};'>Probabilités – IIV3</h4>",
#                     unsafe_allow_html=True
#                 )
#                 probas_iiv3 = pd.Series(y_pred_iiv3[0], index=classes_iiv3).sort_values(ascending=False)
#                 st.bar_chart(probas_iiv3)
# ----------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET 3 : PREDICTIONS CNN vs IIV3
# ----------------------------------------------------
with tab3:
    st.header("COMPARAISON DES PRÉDICTIONS : BASELINE CNN VS INCEPTIONV3")
    iiv3_model = load_model_iiv3()
    # Libérer ICNT si présent
    if "icnt_model" in globals():
        st.write("ICNT était présent, on enleve")
        del icnt_model
        gc.collect()

    # --- Palette harmonisée
    model_colors = {
        "BASELINE CNN": "#4E79A7",   # Bleu doux/grisé
        "IIV3": "#A67C7C"            # Corail feutré
    }
    import psutil
    st.write(f"RAM utilisée : {psutil.virtual_memory().percent}%")


    # --- BLOC 1 : SELECTION IMAGE
    st.markdown(
        """
        <div style="border:2px solid #5A2D82; border-radius:8px;
                    padding:12px; background-color:#F9F6FB; margin:20px 0;">
            <div style="font-size:20px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                Sélectionnez une image du blind test pour comparer les prédictions des deux modèles.
            </div>
        """,
        unsafe_allow_html=True
    )
    df_blind = load_blind_test_sample()

    # Créer une colonne "label" plus lisible
    df_blind["label"] = df_blind["source_path"].apply(lambda x: x.split("/")[-1])

    # Selectbox avec labels lisibles
    selected_label = st.selectbox(
        "Image du blind test",
        df_blind["label"].tolist(),
        key="selectbox_cnn_vs_iiv3"
    )

    # Récupérer l’URL correspondant au label choisi
    selected_url = df_blind.loc[df_blind["label"] == selected_label, "source_path"].values[0]

    if selected_url:
        img = Image.open(requests.get(selected_url, stream=True).raw).convert("RGB")
        img = img.resize((250, 250))
        st.image(img, caption="Image sélectionnée", use_column_width=False)

        import altair as alt
        # --- Prétraitement
        img_batch_cnn = preprocess_image_cnn(img, target_size=(227, 227))
        img_batch_iiv3 = preprocess_image_iiv3(img, target_size=(224, 224))

        # --- BLOC 2 : CHOIX DES MODELES
        st.markdown(
            """
            <div style="border:2px solid #5A2D82; border-radius:8px;
                        padding:12px; background-color:#F9F6FB; margin:20px 0;">
                <div style="font-size:20px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                    ⚙️ Choisissez les modèles à comparer
                </div>
            """,
            unsafe_allow_html=True
        )
        cb_col1, cb_col2 = st.columns(2)
        with cb_col1:
            st.markdown(
                f"<h5 style='color:{model_colors['BASELINE CNN']}; font-size:18px;'>BASELINE CNN</h5>",
                unsafe_allow_html=True
            )
            use_baseline = st.checkbox("PREDIRE AVEC BASELINE CNN", value=True, key="checkbox_baseline_iiv3")
        with cb_col2:
            st.markdown(
                f"<h5 style='color:{model_colors['IIV3']}; font-size:18px;'>INCEPTIONV3</h5>",
                unsafe_allow_html=True
            )
            use_iiv3 = st.checkbox("PREDIRE AVEC INCEPTIONV3", value=True, key="checkbox_iiv3")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- BLOC 3 : PREDICTIONS
        row2_col1, row2_col2 = st.columns(2)
        pred_base, pred_iiv3 = None, None

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
                    <div style="border:2px solid {model_colors['BASELINE CNN']};
                                border-radius:8px; padding:12px; background:#FAFAFA;">
                        <h5 style="color:{model_colors['BASELINE CNN']}; text-align:center; margin:0;">
                            CLASSE PRÉDITE : {pred_base.upper()}
                        </h5>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        if use_iiv3:
            try:
                # iiv3_model = load_model_iiv3()
                y_pred_iiv3 = iiv3_model.predict(img_batch_iiv3)
                pred_iiv3 = classes_iiv3[np.argmax(y_pred_iiv3)]
        
                with row2_col2:
                    st.markdown(
                        f"<h4 style='color:{model_colors['IIV3']};'>PRÉDICTION – INCEPTIONV3</h4>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"""
                        <div style="border:2px solid {model_colors['IIV3']};
                                    border-radius:8px; padding:12px; background:#FAFAFA;">
                            <h5 style="color:{model_colors['IIV3']}; text-align:center; margin:0;">
                                CLASSE PRÉDITE : {pred_iiv3.upper()}
                            </h5>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
               
                    
            except Exception as e:
                with row2_col2:
                    st.warning(f"Erreur de chargement du modèle IIV3 : {e}")

        # --- BLOC 4 : PROBABILITES
        row3_col1, row3_col2 = st.columns(2)
        if use_baseline:
            probas_base = pd.Series(y_pred_base[0], index=classes_cnn).sort_values(ascending=False)
            df_base = probas_base.reset_index()
            df_base.columns = ["Classe", "Probabilité"]
            with row3_col1:
                st.markdown(
                    f"<h4 style='color:{model_colors['BASELINE CNN']};'> PROBABILITÉS – BASELINE CNN</h4>",
                    unsafe_allow_html=True
                )
                chart_base = alt.Chart(df_base).mark_bar(color=model_colors["BASELINE CNN"]).encode(
                    x=alt.X("Classe", title="CLASSE"),
                    y=alt.Y("Probabilité", title="PROBABILITÉ")
                ).properties(height=300)
                st.altair_chart(chart_base, use_container_width=True)
            if "y_pred_base" in locals():
            # Libération mémoire après affichage
                del y_pred_base, probas_base, df_base, chart_base
                gc.collect()

        if use_iiv3:
            probs_iiv3 = y_pred_iiv3[0]
            probas_iiv3 = pd.Series(probs_iiv3, index=classes_iiv3).sort_values(ascending=False)
            df_iiv3 = probas_iiv3.reset_index()
            df_iiv3.columns = ["Classe", "Probabilité"]
            with row3_col2:
                st.markdown(
                    f"<h4 style='color:{model_colors['IIV3']};'> PROBABILITÉS – INCEPTIONV3</h4>",
                    unsafe_allow_html=True
                )
                chart_iiv3 = alt.Chart(df_iiv3).mark_bar(color=model_colors["IIV3"]).encode(
                    x=alt.X("Classe", title="CLASSE"),
                    y=alt.Y("Probabilité", title="PROBABILITÉ")
                ).properties(height=300)
                st.altair_chart(chart_iiv3, use_container_width=True)
                 # ---ON LIBERE LA MEMOIRE APRES PREDICTION DE LA CLASSE:
                if "y_pred_iiv3" in locals():
                    del y_pred_iiv3, probs_iiv3, probas_iiv3, df_iiv3, chart_iiv3
                    gc.collect()

        


    # ----------------------------------------------------
    # COMPOSANT GRAPHIQUE ONGLET 4 : APERÇU DES PERFORMANCES
    # ----------------------------------------------------
    with tab4:
        st.header("APERÇU DES PERFORMANCES")
    
        # --------------------------------------------
        # --- Sélecteur dynamique avec libellé custom
        # --------------------------------------------
        # Dictionnaire de correspondance valeur technique -> label lisible
        model_labels = {
            "baseline_cnn": "BASELINE CNN",
            "icnt": "IMPROVED CONVNEXT-TINY",
            "iiv3": "IMPROVED INCEPTIONV3"
        }
        st.markdown(
            """
            <div style="font-size:18px; font-weight:600; color:#005A9C; margin-bottom:12px;margin-top:50px;">
                Choisissez un modèle à analyser
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("BASELINE CNN", key="btn_cnn"):
                st.session_state["perf_select"] = "baseline_cnn"
        with col2:
            if st.button("IMPROVED CONVNEXT-TINY", key="btn_icnt"):
                st.session_state["perf_select"] = "icnt"
        with col3:
            if st.button("IMPROVED INCEPTIONV3", key="btn_iiv3"):
                st.session_state["perf_select"] = "iiv3"
    
        selected_model = st.session_state.get("perf_select", "baseline_cnn")
        res = HF_PERFORMANCES[selected_model]
        # Indicateur visuel du modèle choisi
        st.markdown(
            f"""
            <div style="margin-top:12px; font-size:16px; font-weight:600; color:#005A9C;">
                ✅ Modèle sélectionné : <span style="color:#222;">{model_labels[selected_model]}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
        # ---------------------------------
        # --- Bloc 1 : METRIQUES GLOBALES
        # ---------------------------------
        st.markdown(
            """
            <div style="border:2px solid #5A2D82; border-radius:8px; padding:12px; background-color:#F9F6FB; margin:20px 0;">
                <div style="font-size:22px; font-weight:600; color:#5A2D82; margin-bottom:12px;">
                    METRIQUES GLOBALES
                </div>
            """,
            unsafe_allow_html=True
        )
        metrics_df = pd.read_csv(res["metrics"])
        if "model_path" in metrics_df.columns:
            metrics_df = metrics_df.drop(columns=["model_path"])
        styled_df = metrics_df.style.set_table_styles([
            {'selector': 'th', 'props': [('font-size', '18pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
            {'selector': 'td', 'props': [('font-size', '16pt'), ('text-align', 'center')]}
        ])
        st.markdown(styled_df.to_html(), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
        # -------------------------------------
        # --- Bloc 2 : APPRENTISSAGE DU MODÈLE
        # -------------------------------------
        st.markdown(
            """
            <div style="border:2px solid #5A2D82; border-radius:8px; padding:12px; background-color:#F9F6FB; margin:20px 0;">
                <div style="font-size:22px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                    APPRENTISSAGE DU MODÈLE
                </div>
                <div style="font-size:20px; color:#444; margin-bottom:12px;">
                    Ces courbes montrent la progression de l'entraînement et permettent de vérifier la convergence.
                </div>
            """,
            unsafe_allow_html=True
        )
        col_left, col_center, col_right = st.columns([2,6,2])
        with col_center:
            st.image(res["learning_curves"], caption="Courbes Loss & Accuracy")
        st.markdown("</div>", unsafe_allow_html=True)
    
        # ---------------------------------------------------------
        # --- Bloc 3 : REPARTITION DES PREDICTIONS DANS LES CLASSES
        # ---------------------------------------------------------
        st.markdown(
            """
            <div style="border:2px solid #5A2D82; border-radius:8px; padding:12px; background-color:#F9F6FB; margin:20px 0;">
                <div style="font-size:22px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                    RÉPARTITION DES PRÉDICTIONS DANS LES CLASSES
                </div>
            """,
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
            st.image(res["confusion_matrix"], caption="MATRICE DE CONFUSION")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
            st.image(res["roc_curve"], caption="COURBES ROC - AUC")
            st.markdown("</div>", unsafe_allow_html=True)
    
        report_df = pd.read_csv(res["classification_report"])
        st.markdown(
            """
            <div style="color:#005A9C; font-weight:600; font-size:16px; margin-bottom:6px;">
                RAPPORT DE CLASSIFICATION DÉTAILLÉ
            </div>
            """,
            unsafe_allow_html=True
        )
        with st.expander("Voir le rapport détaillé", expanded=False):
            st.dataframe(report_df, use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
    
        # -----------------------------------------------------
        # --- Bloc 4 : SEPARABILITE DES CLASSES
        # -----------------------------------------------------
        st.markdown(
            """
            <div style="border:2px solid #5A2D82; border-radius:8px; padding:12px; background-color:#F9F6FB; margin:20px 0;">
                <div style="font-size:22px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                    SÉPARABILITÉ DES CLASSES
                </div>
                <div style="font-size:20px; color:#444; margin-bottom:12px;">
                    Cette projection PCA en 3D permet de visualiser la séparation des classes à partir des features extraits par le modèle.
                </div>
            """,
            unsafe_allow_html=True
        )
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            st.components.v1.html(requests.get(res["pca"]).text, height=490)
            st.markdown("<div style='text-align:center; font-size:16px; font-weight:600; color:#005A9C;'>Vue A</div>", unsafe_allow_html=True)
        with row1_col2:
            st.components.v1.html(requests.get(res["pca"]).text, height=490)
            st.markdown("<div style='text-align:center; font-size:16px; font-weight:600; color:#005A9C;'>Vue B</div>", unsafe_allow_html=True)
        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            st.components.v1.html(requests.get(res["pca"]).text, height=490)
            st.markdown("<div style='text-align:center; font-size:16px; font-weight:600; color:#005A9C;'>Vue C</div>", unsafe_allow_html=True)
        with row2_col2:
            st.components.v1.html(requests.get(res["pca"]).text, height=490)
            st.markdown("<div style='text-align:center; font-size:16px; font-weight:600; color:#005A9C;'>Vue D</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
        # -----------------------------------------------------
        # --- Bloc 5 : EXPLICABILITE
        # -----------------------------------------------------
        st.markdown(
            """
            <div style="border:2px solid #5A2D82; border-radius:8px; padding:12px; background-color:#F9F6FB;margin:20px 0 70px 0;">
                <div style="font-size:22px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                    EXPLICABILITÉ
                </div>
                <div style="font-size:20px; color:#444; margin-bottom:20px;">
                    Les visualisations GradCAM ci-dessous illustrent les zones activées par le modèle lors de ses prédictions.
                </div>
            """,
            unsafe_allow_html=True
        )
        col_left, col_center, col_right = st.columns([2,6,2])
        with col_center:
            grad_col1, grad_col2 = st.columns([1,1])
            with grad_col1:
                st.image(res["gradcam_success"], width=400)
                st.markdown(
                    "<div style='text-align:left; font-size:18px; font-weight:600; color:#444;'>GradCAM - prédiction correcte</div>",
                    unsafe_allow_html=True
                )
            with grad_col2:
                st.image(res["gradcam_error"], width=400)
                st.markdown(
                    "<div style='text-align:left; font-size:18px; font-weight:600; color:#444;'>GradCAM - prédiction en erreur</div>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)    
        # ----------------------------------------------
        # --- Bloc 5 : ARCHITECTURE DU MODELE technique
        # -----------------------------------------------
        st.markdown(
        """
        <div style="font-size:22px; font-weight:600; color:#005A9C; margin-top:16px;">
            DESIGN DU MODÈLE
        </div>
        """,
        unsafe_allow_html=True
        )
        with st.expander("Voir le design détaillé"):
            summary_df = pd.read_csv(res["summary"])
            st.dataframe(summary_df, use_container_width=False)


# ----------------------------------------------------
# COMPOSANT GRAPHIQUE ONGLET 5 : COMPARAISON MODELES
# ----------------------------------------------------
with tab5:
    st.header("COMPARAISON DES MODÈLES")

    # ---------------------------------
    # --- Bloc 1 : Métriques globales
    # ---------------------------------
    st.markdown(
        """
        <div style="border:2px solid #5A2D82; border-radius:8px;
                    padding:12px; background-color:#F9F6FB; margin:20px 0;">
            <div style="font-size:22px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                MÉTRIQUES GLOBALES
            </div>
        """,
        unsafe_allow_html=True
    )
    
    metrics_df = pd.read_csv(HF_COMPARAISON["metrics"]["csv"])
    # --- Suppression de la colonne inutile
    if "model_path" in metrics_df.columns:
        metrics_df = metrics_df.drop(columns=["model_path"])
    
    # --- Styliser le tableau avec pandas Styler
    styled_df = metrics_df.style.set_table_styles([
        {'selector': 'th', 'props': [('font-size', '18pt'), ('font-weight', 'bold'), ('text-align', 'center')]},
        {'selector': 'td', 'props': [('font-size', '16pt'), ('text-align', 'center')]}
    ])
    
    # --- Affichage en HTML
    st.markdown(styled_df.to_html(), unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


    # ---------------------------------
    # --- Bloc 2 : Scatter plots multi-métriques
    # ---------------------------------
    st.markdown(
        """
        <div style="border:2px solid #5A2D82; border-radius:8px;
                    padding:12px; background-color:#F9F6FB; margin:20px 0;">
            <div style="font-size:22px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                SCATTER PLOTS DE COMPARAISON
            </div>
        """,
        unsafe_allow_html=True
    )
    st.image(HF_COMPARAISON["metrics"]["scatter"])
    st.markdown("<div style='text-align:center; font-size:18px; font-weight:600;'>Comparaison par couples de métriques</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------------------------------
    # --- Bloc 3 : Radar performances globales
    # ---------------------------------
    st.markdown(
        """
        <div style="border:2px solid #5A2D82; border-radius:8px;
                    padding:12px; background-color:#F9F6FB; margin:20px 0;">
            <div style="font-size:22px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                🕸️ PERFORMANCES GLOBALES (valeurs brutes)
            </div>
        """,
        unsafe_allow_html=True
    )
    
    # Création de colonnes pour centrer l'image
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(HF_COMPARAISON["metrics"]["radar_perf"])
        st.markdown(
            "<div style='text-align:center; font-size:16px; font-weight:600;'>Radar plot des performances globales</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)


    # ---------------------------------
    # --- Bloc 4 : Scatter + Barplots (équilibre inter-classes)
    # ---------------------------------
    st.markdown(
        """
        <div style="border:2px solid #5A2D82; border-radius:8px;
                    padding:12px; background-color:#F9F6FB; margin:20px 0;">
            <div style="font-size:22px; font-weight:600; color:#5A2D82; margin-bottom:8px;">
                ÉQUILIBRE INTER-CLASSES : F1_mean vs Recall_mean & DISPERSION
            </div>
            <div style="font-size:16px; color:#444; margin-bottom:12px;">
                Le scatter plot suggère que ICNT est meilleur que le baseline CNN. 
                Mais les barplots avec barres d'erreur révèlent une dispersion plus forte, 
                donc une équité moindre.
            </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([2,2,2])

    with col1:
        st.image(HF_COMPARAISON["equilibre"]["scatter"], width=480)
        st.markdown("<div style='text-align:center; font-size:16px; font-weight:600;'>Scatter F1_mean vs Recall_mean</div>", unsafe_allow_html=True)

    with col2:
        st.image(HF_COMPARAISON["equilibre"]["bar_recall"])
        st.markdown("<div style='text-align:center; font-size:16px; font-weight:600;'>Recall moyen ± écart-type</div>", unsafe_allow_html=True)

    with col3:
        st.image(HF_COMPARAISON["equilibre"]["bar_f1"])
        st.markdown("<div style='text-align:center; font-size:16px; font-weight:600;'>F1 moyen ± écart-type</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)






