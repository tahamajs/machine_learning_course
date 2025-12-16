import argparse
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import warnings
import mlflow
from src.mlflow_tracking import log_classic_experiment, log_lstm_history

warnings.filterwarnings("ignore")

# Configuration
DATASET_PATH = "data"  # Folder containing subfolders of Dastgahs (change as needed)
SAMPLE_RATE = 22050
DURATION = 30  # seconds to load from each clip
N_MFCC = 13    # Number of MFCCs
MAX_SEQ_LEN = 1300  # Max time-steps for LSTM sequences (pad/truncate)


def extract_features(file_path):
    """
    Extract fixed-size features for classic ML models.
    Returns a 1D numpy array (mfcc_mean, mfcc_std, zcr).
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        zcr = np.array([np.mean(librosa.feature.zero_crossing_rate(audio))])

        return np.hstack([mfccs_mean, mfccs_std, zcr])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def extract_sequential_features(file_path, max_len=MAX_SEQ_LEN):
    """
    Extract sequential MFCCs for RNN/LSTM input. Returns shape (time_steps, features).
    """
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfccs = mfccs.T  # (time, features)

        if len(mfccs) < max_len:
            pad_width = max_len - len(mfccs)
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode="constant")
        else:
            mfccs = mfccs[:max_len, :]
        return mfccs
    except Exception as e:
        print(f"Error processing (seq) {file_path}: {e}")
        return None


def load_data(dataset_path=DATASET_PATH):
    features_classic = []
    features_seq = []
    labels = []
    file_paths = []

    for label in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, label)
        if not os.path.isdir(folder_path):
            continue
        print(f"Processing Dastgah: {label}...")
        for file in sorted(os.listdir(folder_path)):
            if not file.lower().endswith((".mp3", ".wav", ".flac")):
                continue
            file_path = os.path.join(folder_path, file)
            classic = extract_features(file_path)
            seq = extract_sequential_features(file_path)
            if classic is None or seq is None:
                continue
            features_classic.append(classic)
            features_seq.append(seq)
            labels.append(label)
            file_paths.append(file_path)

    X_classic = np.array(features_classic)
    X_seq = np.array(features_seq)
    y_raw = np.array(labels)
    return X_classic, X_seq, y_raw, file_paths


def visualize_audio(file_path):
    y, sr = librosa.load(file_path, duration=10)
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Time Domain: Waveform")

    plt.subplot(1, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title("Frequency Domain: Spectrogram")
    plt.tight_layout()
    plt.show()


def train_and_evaluate(X_classic, X_seq, y, class_names):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_classic, y, test_size=0.25, random_state=42, stratify=y)
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y, test_size=0.25, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    trained_models = {}

    # KNN
    print("\n--- Training KNN ---")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    y_pred_knn = knn.predict(X_test_scaled)
    results["KNN"] = {"f1": f1_score(y_test, y_pred_knn, average="weighted"), "y_pred": y_pred_knn}
    trained_models["knn"] = knn

    # SVM
    print("\n--- Training SVM ---")
    svm = SVC(kernel="rbf", C=1.0, probability=True)
    svm.fit(X_train_scaled, y_train)
    y_pred_svm = svm.predict(X_test_scaled)
    results["SVM"] = {"f1": f1_score(y_test, y_pred_svm, average="weighted"), "y_pred": y_pred_svm}
    trained_models["svm"] = svm

    # RandomForest (optional baseline)
    print("\n--- Training RandomForest (baseline) ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results["RandomForest"] = {"f1": f1_score(y_test, y_pred_rf, average="weighted"), "y_pred": y_pred_rf}
    trained_models["rf"] = rf

    # LSTM
    print("\n--- Training LSTM ---")
    y_seq_train_cat = to_categorical(y_seq_train)
    y_seq_test_cat = to_categorical(y_seq_test)

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_seq.shape[1], X_seq.shape[2]), return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(len(class_names), activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Callbacks: early stopping and checkpoint
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    # Temporary checkpoint path in memory; caller can move it
    checkpoint_path = "lstm_best.h5"
    mc = ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True, verbose=0)

    history = model.fit(X_seq_train, y_seq_train_cat, epochs=60, batch_size=16, validation_data=(X_seq_test, y_seq_test_cat), callbacks=[es, mc], verbose=1)
    loss, acc = model.evaluate(X_seq_test, y_seq_test_cat, verbose=0)
    y_pred_lstm_prob = model.predict(X_seq_test)
    y_pred_lstm = np.argmax(y_pred_lstm_prob, axis=1)
    results["LSTM"] = {"f1": f1_score(y_seq_test, y_pred_lstm, average="weighted"), "y_pred": y_pred_lstm, "history": history}
    trained_models["lstm"] = model

    # Save scaler and label encoder will be done by caller
    trained_models["scaler"] = scaler

    # Print summary
    for name, info in results.items():
        print(f"{name}: f1={info['f1']:.4f}")

    return results, trained_models, (X_train_scaled, X_test_scaled, y_train, y_test), (X_seq_train, X_seq_test, y_seq_train, y_seq_test)


def report_results(model_name, y_true, y_pred, class_names):
    print(f"\nResults for {model_name}:")
    print("-" * 30)
    print(classification_report(y_true, y_pred, target_names=class_names))
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


def perform_clustering(X_scaled, y_all, class_names, ks=(7, 20)):
    X_cluster = X_scaled
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_cluster)

    for k in ks:
        print(f"\nClustering with k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X_cluster)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap="viridis", s=50, alpha=0.6)
        plt.title(f"K-Means Clustering (k={k})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter)
        plt.show()

        df_cluster = pd.DataFrame({"True_Label": class_names[y_all], "Cluster": clusters})
        print(pd.crosstab(df_cluster["True_Label"], df_cluster["Cluster"]))


def main():
    parser = argparse.ArgumentParser(description="Full pipeline: feature extraction, train, save models, clustering")
    parser.add_argument("--data_dir", default="data", help="Dataset directory with subfolders per class")
    parser.add_argument("--save_models_dir", default=None, help="If set, save trained models and scalers to this directory")
    parser.add_argument("--no_visuals", action="store_true", help="Disable plotting (useful for headless runs)")
    parser.add_argument("--mlflow", action="store_true", help="Log run to MLflow")
    parser.add_argument("--mlflow_uri", default=None, help="MLflow tracking URI (e.g. http://localhost:5000)")
    parser.add_argument("--mlflow_experiment", default="final_project", help="MLflow experiment name")
    args = parser.parse_args()

    X_classic, X_seq, y_raw, file_paths = load_data(dataset_path=args.data_dir)
    if len(y_raw) == 0:
        print("No data found. Please populate the dataset folder and rerun.")
        return

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    print(f"Loaded {len(y)} items. Classes: {class_names}")

    results, trained_models, classic_split, seq_split = train_and_evaluate(X_classic, X_seq, y, class_names)

    # Configure MLflow if requested
    if args.mlflow:
        if args.mlflow_uri:
            mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        run = mlflow.start_run()
        mlflow.log_param("data_dir", args.data_dir)
        mlflow.log_param("num_samples", len(y))
        mlflow.log_param("classes", ",".join(class_names))
    # Detailed reports
    for name, info in results.items():
        if name == "LSTM":
            y_true = seq_split[3]
            y_pred = info["y_pred"]
            if not args.no_visuals:
                report_results(name, y_true, y_pred, class_names)
                hist = info["history"].history
                plt.plot(hist["accuracy"], label="Train Accuracy")
                plt.plot(hist["val_accuracy"], label="Val Accuracy")
                plt.title("LSTM Training History")
                plt.legend()
                plt.show()
        else:
            y_true = classic_split[3]
            y_pred = info["y_pred"]
            if not args.no_visuals:
                report_results(name, y_true, y_pred, class_names)

    # Clustering (use scaled classic features: combine train+test for clustering)
    X_train_scaled, X_test_scaled, y_train, y_test = classic_split
    X_scaled_all = np.vstack([X_train_scaled, X_test_scaled])
    y_all = np.hstack([y_train, y_test])
    if not args.no_visuals:
        perform_clustering(X_scaled_all, y_all, class_names, ks=(7, 20))

    # Save models and preprocessing artifacts
    if args.save_models_dir:
        os.makedirs(args.save_models_dir, exist_ok=True)
        # Save classical models
        if "knn" in trained_models:
            joblib.dump(trained_models["knn"], os.path.join(args.save_models_dir, "knn.joblib"))
        if "svm" in trained_models:
            joblib.dump(trained_models["svm"], os.path.join(args.save_models_dir, "svm.joblib"))
        if "rf" in trained_models:
            joblib.dump(trained_models["rf"], os.path.join(args.save_models_dir, "rf.joblib"))
        # Save scaler and label encoder
        if "scaler" in trained_models:
            joblib.dump(trained_models["scaler"], os.path.join(args.save_models_dir, "scaler.joblib"))
        joblib.dump(le, os.path.join(args.save_models_dir, "label_encoder.joblib"))
        # Save LSTM (Keras)
        if "lstm" in trained_models:
            lstm_path = os.path.join(args.save_models_dir, "lstm_model.h5")
            trained_models["lstm"].save(lstm_path)
        print(f"Saved models and artifacts to {args.save_models_dir}")

    # Log to MLflow
    if args.mlflow:
        # Log classical models metrics
        artifacts_dir = args.save_models_dir if args.save_models_dir else None
        for name, info in results.items():
            params = {"model": name}
            metrics = {"f1_weighted": float(info["f1"])}
            # save confusion matrix as artifact if visuals enabled
            if not args.no_visuals:
                try:
                    # plot and save cm
                    if name == "LSTM":
                        y_t = seq_split[3]
                        y_p = info["y_pred"]
                    else:
                        y_t = classic_split[3]
                        y_p = info["y_pred"]
                    cm = confusion_matrix(y_t, y_p)
                    import matplotlib
                    matplotlib.use("Agg")
                    plt.figure(figsize=(6,5))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
                    plt.title(f"Confusion Matrix - {name}")
                    cm_path = os.path.join("tmp_cm.png")
                    plt.savefig(cm_path)
                    plt.close()
                    if artifacts_dir:
                        try:
                            os.replace(cm_path, os.path.join(artifacts_dir, f"cm_{name}.png"))
                        except Exception:
                            pass
                except Exception:
                    pass
            # Log using helper
            try:
                log_classic_experiment(f"{args.mlflow_experiment}_{name}", params, metrics, artifacts_dir=artifacts_dir)
            except Exception:
                # fallback to mlflow direct logging
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

        # Log LSTM history
        if "LSTM" in results and not args.no_visuals:
            try:
                hist = results["LSTM"]["history"]
            except Exception:
                hist = None
            if hist is not None:
                log_lstm_history(args.mlflow_experiment + "_LSTM", hist, artifacts_dir=artifacts_dir)
        mlflow.end_run()


if __name__ == "__main__":
    main()


