# Who's Speaking

Who's Speaking is an explainable speaker-recognition web app for a speech-processing course. It avoids black-box deep learning and instead uses interpretable acoustic features plus an SVM classifier.

## Core Pipeline

- Training input: one `.zip` archive up to 100 MB
- Labels: speaker folder names inside the archive
- Supported audio: `.wav`, `.mp3`, `.flac`
- Feature groups:
  - short-term average energy
  - zero crossing rate
  - 12th-order LPC coefficients
  - mean log-mel energies from the first 20 mel bands
- Normalization: `StandardScaler`
- Classifier: `SVC(kernel="rbf", probability=True)`
- Open-set logic:
  - reject if top probability is below 60%
  - reject if the sample falls outside the predicted speaker's training neighborhood

## Automatic Segmentation

Long recordings are no longer treated as a single training example.

During training, each audio file is:

1. Loaded at `16 kHz` mono
2. Framed with a `25 ms` window and `10 ms` hop
3. Gated by short-term energy to keep speech-active regions
4. Split into voiced segments of about `3 seconds` with `1.5 second` hop
5. Converted into one 36-dimensional feature vector per segment

This means one 2-4 minute clip can generate many training samples for the same speaker.

Prediction also uses the same segmentation logic and averages segment-level class probabilities before making the final speaker decision.

## Project Layout

- `backend/`
  - `app.py`: Flask routes
  - `audio_pipeline.py`: segmentation, feature extraction, training, prediction
- `frontend/`
  - React + Vite + Tailwind UI

## Backend Behavior

### `POST /train`

- Deletes previous uploads, extracted data, and saved models
- Safely extracts the uploaded archive
- Discovers audio files from speaker folders
- Segments long recordings into voiced windows
- Trains the scaler, label encoder, and SVM
- Saves:
  - `model.joblib`
  - `scaler.joblib`
  - `encoder.joblib`
  - `training_features.npy`
  - `training_labels.npy`
  - `training_manifest.json`
  - `feature_schema.json`

### `POST /predict`

- Accepts one test audio file
- Applies the same segmentation + feature pipeline
- Averages segment-level SVM probabilities
- Returns either the predicted speaker or `I don't recognize you.`

## Local Setup

### Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

The Vite dev server proxies `/health`, `/train`, and `/predict` to `http://127.0.0.1:5000`.
