from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import librosa
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac"}
MODEL_FILENAME = "model.joblib"
SCALER_FILENAME = "scaler.joblib"
ENCODER_FILENAME = "encoder.joblib"
FEATURES_FILENAME = "training_features.npy"
LABELS_FILENAME = "training_labels.npy"
MANIFEST_FILENAME = "training_manifest.json"
SCHEMA_FILENAME = "feature_schema.json"
PROBABILITY_THRESHOLD = 0.60


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    frame_duration: float = 0.025
    hop_duration: float = 0.010
    lpc_order: int = 12
    n_mels: int = 20

    @property
    def frame_length(self) -> int:
        return int(self.sample_rate * self.frame_duration)

    @property
    def hop_length(self) -> int:
        return int(self.sample_rate * self.hop_duration)


CONFIG = AudioConfig()

FEATURE_SCHEMA = {
    "short_term_energy": ["energy_mean", "energy_std"],
    "zero_crossing_rate": ["zcr_mean", "zcr_std"],
    "linear_prediction_coefficients": [
        f"lpc_{index}" for index in range(1, CONFIG.lpc_order + 1)
    ],
    "mel_spectrogram": [f"mel_{index}" for index in range(1, CONFIG.n_mels + 1)],
}
FEATURE_NAMES = [
    *FEATURE_SCHEMA["short_term_energy"],
    *FEATURE_SCHEMA["zero_crossing_rate"],
    *FEATURE_SCHEMA["linear_prediction_coefficients"],
    *FEATURE_SCHEMA["mel_spectrogram"],
]


def feature_dimension() -> int:
    return len(FEATURE_NAMES)


def _normalize_signal(signal: np.ndarray) -> np.ndarray:
    if signal.size == 0:
        raise ValueError("Audio signal is empty.")

    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak
    return signal.astype(np.float32)


def _frame_signal(signal: np.ndarray) -> np.ndarray:
    frame_length = CONFIG.frame_length
    hop_length = CONFIG.hop_length

    if signal.size < frame_length:
        signal = np.pad(signal, (0, frame_length - signal.size), mode="constant")

    return librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length)


def _short_term_energy(frames: np.ndarray) -> np.ndarray:
    return np.mean(np.square(frames), axis=0)


def _zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    signs = np.sign(frames)
    signs[signs == 0] = -1
    sign_changes = np.abs(np.diff(signs, axis=0))
    return np.sum(sign_changes, axis=0) / (2.0 * frames.shape[0])


def _valid_lpc_frames(frames: np.ndarray, energies: np.ndarray) -> np.ndarray:
    energy_floor = max(1e-6, float(np.percentile(energies, 35)))
    active_mask = energies >= energy_floor
    active_frames = frames[:, active_mask]
    if active_frames.size == 0:
        active_frames = frames
    return active_frames


def _lpc_summary(frames: np.ndarray, energies: np.ndarray) -> np.ndarray:
    active_frames = _valid_lpc_frames(frames, energies)
    coefficients: list[np.ndarray] = []

    for index in range(active_frames.shape[1]):
        frame = active_frames[:, index]
        if np.allclose(frame, 0.0):
            continue

        try:
            lpc_coefficients = librosa.lpc(frame, order=CONFIG.lpc_order)
            coefficients.append(np.asarray(lpc_coefficients[1:], dtype=np.float32))
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue

    if not coefficients:
        return np.zeros(CONFIG.lpc_order, dtype=np.float32)

    return np.mean(np.vstack(coefficients), axis=0).astype(np.float32)


def _mel_summary(signal: np.ndarray) -> np.ndarray:
    mel_spectrogram = librosa.feature.melspectrogram(
        y=signal,
        sr=CONFIG.sample_rate,
        n_fft=CONFIG.frame_length,
        hop_length=CONFIG.hop_length,
        n_mels=CONFIG.n_mels,
        power=2.0,
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return np.mean(log_mel_spectrogram, axis=1).astype(np.float32)


def extract_feature_vector(audio_path: Path | str) -> np.ndarray:
    signal, _ = librosa.load(audio_path, sr=CONFIG.sample_rate, mono=True)
    signal = _normalize_signal(signal)
    frames = _frame_signal(signal)

    energies = _short_term_energy(frames)
    zcr = _zero_crossing_rate(frames)
    lpc = _lpc_summary(frames, energies)
    mel = _mel_summary(signal)

    feature_vector = np.concatenate(
        [
            np.array([np.mean(energies), np.std(energies)], dtype=np.float32),
            np.array([np.mean(zcr), np.std(zcr)], dtype=np.float32),
            lpc,
            mel,
        ]
    )
    return feature_vector


def save_feature_schema(artifact_dir: Path) -> None:
    schema_payload = {
        "sample_rate_hz": CONFIG.sample_rate,
        "frame_duration_seconds": CONFIG.frame_duration,
        "hop_duration_seconds": CONFIG.hop_duration,
        "lpc_order": CONFIG.lpc_order,
        "mel_filter_banks": CONFIG.n_mels,
        "feature_names": FEATURE_NAMES,
        "feature_groups": FEATURE_SCHEMA,
    }
    (artifact_dir / SCHEMA_FILENAME).write_text(
        json.dumps(schema_payload, indent=2), encoding="utf-8"
    )


def discover_training_files(dataset_root: Path) -> list[tuple[Path, str]]:
    top_level_directories = sorted(
        child for child in dataset_root.iterdir() if child.is_dir() and not child.name.startswith(".")
    )
    top_level_audio = [
        child for child in dataset_root.iterdir() if child.is_file() and child.suffix.lower() in AUDIO_EXTENSIONS
    ]

    if len(top_level_directories) == 1 and not top_level_audio:
        dataset_root = top_level_directories[0]

    training_files: list[tuple[Path, str]] = []
    for audio_path in sorted(dataset_root.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            continue

        relative_parts = audio_path.relative_to(dataset_root).parts
        if len(relative_parts) < 2:
            continue

        speaker_label = relative_parts[0]
        training_files.append((audio_path, speaker_label))

    return training_files


def train_speaker_identifier(dataset_root: Path, artifact_dir: Path) -> dict:
    training_files = discover_training_files(dataset_root)
    if not training_files:
        raise ValueError(
            "No supported audio files were found inside speaker folders. "
            "Use .wav, .mp3, or .flac files grouped by speaker directory."
        )

    speaker_counts: dict[str, int] = {}
    features: list[np.ndarray] = []
    labels: list[str] = []

    for audio_path, speaker_label in training_files:
        feature_vector = extract_feature_vector(audio_path)
        features.append(feature_vector)
        labels.append(speaker_label)
        speaker_counts[speaker_label] = speaker_counts.get(speaker_label, 0) + 1

    if len(speaker_counts) < 2:
        raise ValueError("Training requires at least two speaker folders.")

    feature_matrix = np.vstack(features)
    label_array = np.asarray(labels)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_matrix)

    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(label_array)

    classifier = SVC(kernel="rbf", probability=True)
    classifier.fit(scaled_features, encoded_labels)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, artifact_dir / MODEL_FILENAME)
    joblib.dump(scaler, artifact_dir / SCALER_FILENAME)
    joblib.dump(encoder, artifact_dir / ENCODER_FILENAME)
    np.save(artifact_dir / FEATURES_FILENAME, feature_matrix)
    np.save(artifact_dir / LABELS_FILENAME, label_array)
    save_feature_schema(artifact_dir)

    manifest = {
        "speaker_count": len(speaker_counts),
        "sample_count": int(feature_matrix.shape[0]),
        "feature_dimension": int(feature_matrix.shape[1]),
        "probability_threshold_percent": int(PROBABILITY_THRESHOLD * 100),
        "speakers": [
            {"name": speaker_name, "utterances": utterance_count}
            for speaker_name, utterance_count in sorted(speaker_counts.items())
        ],
        "classifier": "Support Vector Machine (rbf kernel, probability=True)",
        "scaler": "StandardScaler",
    }
    (artifact_dir / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    return manifest


def load_training_manifest(artifact_dir: Path) -> dict:
    manifest_path = artifact_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError("Training manifest is missing. Train the model again.")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def artifacts_exist(artifact_dir: Path) -> bool:
    required = [
        artifact_dir / MODEL_FILENAME,
        artifact_dir / SCALER_FILENAME,
        artifact_dir / ENCODER_FILENAME,
        artifact_dir / FEATURES_FILENAME,
        artifact_dir / LABELS_FILENAME,
        artifact_dir / MANIFEST_FILENAME,
    ]
    return all(path.exists() for path in required)


def _nearest_neighbor_radius(class_vectors: np.ndarray) -> float:
    if class_vectors.shape[0] <= 1:
        return float("inf")

    distance_matrix = pairwise_distances(class_vectors)
    np.fill_diagonal(distance_matrix, np.inf)
    nearest_neighbor_distances = distance_matrix.min(axis=1)

    percentile_radius = float(np.percentile(nearest_neighbor_distances, 95))
    spread_radius = float(
        np.mean(nearest_neighbor_distances) + (2.0 * np.std(nearest_neighbor_distances))
    )
    return max(percentile_radius, spread_radius)


def _query_to_class_distance(query_vector: np.ndarray, class_vectors: np.ndarray) -> float:
    if class_vectors.size == 0:
        return float("inf")

    return float(pairwise_distances(query_vector.reshape(1, -1), class_vectors).min())


def predict_speaker(audio_path: Path, artifact_dir: Path) -> dict:
    if not artifacts_exist(artifact_dir):
        raise FileNotFoundError("Model artifacts are missing. Train the system first.")

    classifier: SVC = joblib.load(artifact_dir / MODEL_FILENAME)
    scaler: StandardScaler = joblib.load(artifact_dir / SCALER_FILENAME)
    encoder: LabelEncoder = joblib.load(artifact_dir / ENCODER_FILENAME)
    training_features = np.load(artifact_dir / FEATURES_FILENAME)
    training_labels = np.load(artifact_dir / LABELS_FILENAME)

    feature_vector = extract_feature_vector(audio_path).reshape(1, -1)
    scaled_feature_vector = scaler.transform(feature_vector)
    scaled_training_features = scaler.transform(training_features)

    probabilities = classifier.predict_proba(scaled_feature_vector)[0]
    predicted_index = int(np.argmax(probabilities))
    predicted_class = int(classifier.classes_[predicted_index])
    confidence = float(probabilities[predicted_index])
    predicted_label = encoder.inverse_transform([predicted_class])[0]
    class_vectors = scaled_training_features[training_labels == predicted_label]
    nearest_distance = _query_to_class_distance(scaled_feature_vector[0], class_vectors)
    speaker_radius = _nearest_neighbor_radius(class_vectors)
    within_speaker_profile = nearest_distance <= speaker_radius
    recognized = confidence >= PROBABILITY_THRESHOLD and within_speaker_profile

    rejection_reason = None
    if confidence < PROBABILITY_THRESHOLD:
        rejection_reason = "low_confidence"
    elif not within_speaker_profile:
        rejection_reason = "outside_speaker_profile"

    return {
        "recognized": recognized,
        "speaker": predicted_label if recognized else None,
        "confidence": round(confidence * 100, 2),
        "threshold": round(PROBABILITY_THRESHOLD * 100, 2),
        "acoustic_distance": round(nearest_distance, 3),
        "speaker_profile_radius": (
            round(speaker_radius, 3) if np.isfinite(speaker_radius) else None
        ),
        "rejection_reason": rejection_reason,
        "message": (
            f"Speaker identified as {predicted_label}."
            if recognized
            else "I don't recognize you."
        ),
    }
