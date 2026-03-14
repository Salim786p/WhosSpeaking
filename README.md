# Who's Speaking

Who's Speaking is an explainable speaker-recognition web application built for a university speech-processing course. Instead of a black-box deep model, it identifies speakers from interpretable acoustic cues:

- Short-term average energy for the speech envelope and silence discrimination
- Zero crossing rate for voiced versus unvoiced behavior
- 12th-order LPC coefficients as a vocal-tract resonance signature
- Mean log-mel energies from the first 20 filter banks
- StandardScaler normalization and an SVM classifier with an `rbf` kernel

## Project Layout

- `frontend/`: React + Vite + Tailwind CSS user interface
- `backend/`: Flask API, feature extraction, training, and open-set prediction logic

## Backend Behavior

- `POST /train`
  - Accepts a `.zip` archive up to 100 MB.
  - Each speaker must have their own folder inside the archive.
  - Supported audio formats are `.wav`, `.mp3`, and `.flac`.
  - Before training, the backend deletes all prior uploads, extracted features, and saved models.
  - Saves `model.joblib`, `scaler.joblib`, and `encoder.joblib` in `backend/artifacts/`.
- `POST /predict`
  - Accepts a single audio file.
  - Uses `predict_proba` for open-set identification.
  - Verifies the sample stays close to the predicted speaker's training neighborhood in standardized feature space.
  - If confidence is below 60%, returns `I don't recognize you.`

## Feature Vector

Each utterance is summarized into a 36-dimensional vector:

- 2 energy statistics: mean and standard deviation
- 2 ZCR statistics: mean and standard deviation
- 12 LPC coefficients
- 20 log-mel filter-bank means

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

The Vite dev server proxies `/health`, `/train`, and `/predict` to the Flask server on `http://127.0.0.1:5000`.
