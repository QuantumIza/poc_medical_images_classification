# ------------------------------
# CONFIGURATION RESSOURCES HF
# ------------------------------

HF_RESOURCES = {
    "models": {
        "cnn": {
            "local": "outputs/models/cnn_model.h5",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/models/cnn_model.h5"
        },
        "ictn": {
            "local": "outputs/models/ictn_model.h5",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/models/ictn_model.h5"
        }
    },
    "labels": {
        "cnn": {
            "local": "outputs/labels/class_labels_cnn.json",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/labels/class_labels_cnn.json"
        },
        "ictn": {
            "local": "outputs/labels/class_labels_icnt.json",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/labels/class_labels_icnt.json"
        }
    },
    "datasets": {
        "stats": {
            "local": "outputs/eda/dataset_stats.csv",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/eda/dataset_stats.csv"
        },
        "full": {
            "local": "outputs/data/full_dataset.csv",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/data/full_dataset.csv"
        },
        "sample": {
            "local": "outputs/data/sample_dataset.csv",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/data/sample_dataset.csv"
        }
    },
    "history": {
        "cnn": {
            "local": "outputs/history/history_cnn.json",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/history/history_cnn.json"
        },
        "ictn": {
            "local": "outputs/history/history_icnt.json",
            "url": "https://huggingface.co/QuantumIza/poc-baseline-cnn/resolve/main/history/history_icnt.json"
        }
    }
}
