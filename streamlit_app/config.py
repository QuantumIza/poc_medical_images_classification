HF_RESOURCES = {
    "models": {
        "baseline_cnn": {
            "local": "src/outputs/baseline_cnn/model/best_model_baseline_cnn.keras",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/model/best_model_baseline_cnn.keras"
        },
        "icnt": {
            "local": "src/outputs/icnt/model/best_model_icnt.keras",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/model/best_model_icnt.keras"
        },
        "iiv3": {
            "local": "src/outputs/iiv3/model/best_model_iiv3.keras",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/model/best_model_iiv3.keras"
        }
    },
    "datasets": {
        "full": {
            "local": "src/outputs/eda/df_full.csv",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/eda/df_full.csv"
        },
        "sample": {
            "local": "data/my_df_sample.csv",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/df_sample.csv"
        },
        "stats": {   
            "local": "src/outputs/eda/dataset_stats.csv",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/eda/dataset_stats.csv"
        },
        "blind_test": {
            "local": "src/outputs/eda/df_blind_test.csv",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/eda/df_blind_test.csv"
        }
    },
    "history": {
        "baseline_cnn": {
            "local": "src/outputs/baseline_cnn/history/history_baseline_cnn.json",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/history/history_baseline_cnn.json"
        },
        "icnt": {
            "local": "src/outputs/icnt/history/history_icnt.json",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/history/history_icnt.json"
        },
        "iiv3": {
            "local": "src/outputs/iiv3/history/history_iiv3.json",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/history/history_iiv3.json"
        }
    }
}
HF_PERFORMANCES = {
    "baseline_cnn": {
        "learning_curves": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/assets/learning_curves_loss_cnn.png",
        "confusion_matrix": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/assets/matrice_confusion_cnn.png",
        "classification_report": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/test_evaluation/classification_report_cnn.csv",
        "roc_curve": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/assets/roc_multiclasse_cnn.png",
        "metrics": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/metrics/metrics_baseline_cnn.csv",
        "summary": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/summary/summary_baseline_cnn.csv",
        "pca": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/assets/pca_3d_interactif_light_cnn.html",
        "gradcam_success": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/convincing_cases/gradcam_convincing_malignant_conf_1.00_idx_129.png",
        "gradcam_error": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/baseline_cnn/critical_errors/gradcam_error_malignant_to_benign_true_malignant_pred_benign_conf_0.95_idx_7234.png"
    },
    "icnt": {
        "learning_curves": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/assets/learning_curves_loss_cnn.png",
        "confusion_matrix": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/assets/matrice_confusion_icnt.png",
        "classification_report": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/test_evaluation/classification_report_icnt.csv",
        "roc_curve": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/assets/roc_multiclasse_icnt.png",
        "metrics": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/metrics/metrics_icnt.csv",
        "summary": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/summary/summary_modele_icnt.csv",
        "pca": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/assets/pca_3d_interactif_light_icnt.html",
        "gradcam_success": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/convincing_cases/gradcam_convincing_malignant_conf_1.00_idx_106.png",
        "gradcam_error": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/icnt/critical_errors/gradcam_error_malignant_to_normal_true_malignant_pred_normal_conf_1.00_idx_5847.png"
    },
    "iiv3": {
        "learning_curves": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/assets/learning_curves_loss_iiv3.png",
        "confusion_matrix": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/assets/matrice_confusion_iiv3.png",
        "classification_report": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/test_evaluation/classification_report_iiv3.csv",
        "roc_curve": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/assets/roc_multiclasse_iiv3.png",
        "metrics": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/metrics/metrics_iiv3.csv",
        "summary": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/summary/summary_modele_iiv3_structured.csv",
        "pca": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/assets/pca_3d_interactif_light_iiv3.html",
        "gradcam_success": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/convincing_cases/gradcam_convincing_malignant_conf_1.00_idx_8.png",
        "gradcam_error": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/iiv3/critical_errors/gradcam_error_malignant_to_benign_true_malignant_pred_benign_conf_0.99_idx_5400.png"
    }
}
HF_COMPARAISON = {
    "apprentissage": {
        "learning_curves": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/apprentissage/learning_curves_comparaison.png"
    },
    "equilibre": {
        "stats": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_equilibre/models_equilibre_stats.csv",
        "scatter": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_equilibre/scatterplot_f1_vs_recall.png",
        "bar_recall": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_equilibre/barplot_recall_mean.png",
        "bar_f1": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_equilibre/barplot_f1_mean.png",
        "matrices_confusion": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_equilibre/matrices_confusion_comparaison.png"
    },
    "metrics": {
        "csv": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_metrics/models_metrics_comparaison.csv",
        "scatter": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_metrics/scatter_plots_comparaison.png",
        "radar_classes": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_metrics/radar_couverture_des_classes_relatif.png",
        "radar_perf": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_metrics/radar_performance_globale_brut.png",
        "radar_overfit": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_metrics/radar_sur-apprentissage_relatif.png",
        "roc_curves": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/comparaison/models_metrics/roc_curves_comparaison.png"
    }
}

