# Who's Speaking

Who's Speaking is an explainable speaker-identification project built with a React frontend and a Flask backend. The backend does not use an end-to-end deep model. Instead, it extracts interpretable acoustic features from each uploaded audio file, trains an SVM classifier, and then applies open-set checks so unknown voices can be rejected instead of being forced into a known class.

This README explains the server in detail, especially what happens after the user uploads the training ZIP file.

## What This Project Does

The system has two phases:

1. Training
   The user uploads a `.zip` archive where each speaker has their own folder of audio files.
2. Prediction
   The user uploads one test audio file, and the backend either identifies the speaker or rejects the sample as unknown.

The feature pipeline is intentionally transparent:

- Short-term average energy
- Short-time zero crossing rate (STZCR / ZCR)
- Linear predictive analysis (LPA) summarized as 12 LPC coefficients
- Mean log-mel filter-bank energies from 20 mel bands
- StandardScaler normalization
- SVM with `rbf` kernel
- Open-set rejection using confidence plus acoustic-distance consistency

## Repository Layout

- `backend/`
  Flask API, feature extraction, model training, and prediction logic.
- `frontend/`
  React + Vite UI for uploading training data and test audio.

Important backend files:

- `backend/app.py`
  API routes and request handling.
- `backend/audio_pipeline.py`
  Signal processing, feature extraction, model training, artifact saving, and prediction logic.

## Runtime Directories and Saved Files

The backend creates these directories at runtime:

- `backend/artifacts/`
  Stores the trained model and metadata.
- `backend/workspace/uploads/`
  Stores uploaded files temporarily.
- `backend/workspace/extracted/`
  Stores the extracted training dataset.

Saved model artifacts:

- `model.joblib`
  The fitted SVM classifier.
- `scaler.joblib`
  The fitted `StandardScaler`.
- `encoder.joblib`
  The fitted `LabelEncoder`.
- `training_features.npy`
  Raw training feature vectors before scaling.
- `training_labels.npy`
  Original speaker labels as strings.
- `training_manifest.json`
  A summary of the trained dataset and model.
- `feature_schema.json`
  Metadata describing the feature vector layout.

## Training Dataset Format

The backend expects a ZIP archive containing speaker folders. The folder name becomes the speaker label.

Example:

```text
dataset.zip
└── training_corpus/
    ├── Alice/
    │   ├── a1.wav
    │   ├── a2.wav
    │   └── a3.mp3
    ├── Bob/
    │   ├── b1.wav
    │   └── b2.flac
    └── Carol/
        ├── c1.wav
        └── c2.wav
```

Notes:

- Supported audio formats are `.wav`, `.mp3`, and `.flac`.
- The backend accepts one wrapper folder at the top level. In the example above, `training_corpus/` is fine.
- Audio files must be inside speaker folders.
- Top-level audio files are ignored during training.
- At least two speaker folders are required.

## End-to-End Server Workflow

### 1. The frontend sends the training ZIP

When the user clicks **Train Model**, the frontend:

- Reads the selected ZIP file from the browser.
- Builds a `FormData` object with the field name `dataset`.
- Sends `POST /train` to the Flask backend.
- Waits for the full training process to finish in the same request.

There is no background worker or queue here. Training is synchronous, so the request stays open until feature extraction and SVM training are finished.

### 2. The backend validates the upload

In `backend/app.py`, the `/train` route first:

- Ensures the runtime directories exist.
- Checks that a file was actually uploaded.
- Sanitizes the filename with `secure_filename(...)`.
- Verifies that the uploaded file ends with `.zip`.
- Enforces a maximum upload size of 100 MB through Flask's `MAX_CONTENT_LENGTH`.

If the file is too large, Flask returns:

- HTTP `413`
- Message: `Upload exceeds the 100 MB limit.`

### 3. Previous training state is deleted

Before training starts, the backend calls `reset_training_state()`.

That function removes:

- `backend/artifacts/`
- `backend/workspace/`

Then it recreates the runtime folders.

This means:

- only one trained model exists at a time
- retraining replaces the previous model completely
- partially stale files from an older run are cleared out first

### 4. The ZIP is saved to disk

The uploaded ZIP is saved in:

- `backend/workspace/uploads/<sanitized_zip_name>`

The backend keeps a copy there after a successful training run as well. It is removed the next time training resets the workspace.

### 5. The ZIP is checked for unsafe paths and extracted

The backend does not blindly extract the archive.

Before extracting, `safe_extract_archive(...)` loops through every ZIP member and checks that the resolved file path stays inside the intended extraction directory.

Why this matters:

- It prevents path traversal attacks such as `../../somewhere_else.txt`.
- This is a common ZIP extraction safety check, sometimes called protection against "Zip Slip".

If any extracted path would escape the target folder, training is aborted with a `400` error.

If the archive is valid, the ZIP is extracted to:

- `backend/workspace/extracted/`

### 6. The backend discovers which files belong to which speaker

The actual training dataset is discovered by `discover_training_files(...)` in `backend/audio_pipeline.py`.

This function does several important things:

1. It looks at the extracted dataset root.
2. If there is exactly one top-level directory and no top-level audio files, it treats that directory as the real dataset root.
3. It recursively scans for supported audio files.
4. For each valid audio file, it takes the first folder name beneath the dataset root as the speaker label.

Example:

```text
training_corpus/Alice/a1.wav  -> label "Alice"
training_corpus/Bob/b1.wav    -> label "Bob"
```

Important hidden detail:

- The code does not rename files.
- The code does not tag filenames with integers.
- Speaker identity comes from folder names.
- Integer class IDs are created later by `LabelEncoder`, not by changing file names.

Also important:

- Files are processed in sorted order.
- Unsupported extensions are skipped.
- Files that are not inside a speaker folder are skipped.

### 7. Each audio file is converted into one feature vector

For every discovered training audio file, the backend calls `extract_feature_vector(...)`.

Each file becomes one 36-dimensional numeric vector.

#### 7.1 Audio loading

The backend loads each file with:

- sample rate forced to `16,000 Hz`
- `mono=True`

Why:

- A fixed sample rate makes every feature comparable across files.
- Mono removes channel-count differences and keeps the feature pipeline simple.

So even if the original file is stereo or uses a different sample rate, the backend converts it into a consistent form before feature extraction.

#### 7.2 Signal normalization

After loading, the signal is peak-normalized:

```text
signal = signal / max(abs(signal))
```

if the peak is greater than zero.

Why:

- It reduces loudness differences caused by recording volume.
- It helps the model focus more on speaker characteristics than on raw amplitude scale.

If the audio is empty, the backend raises `Audio signal is empty.`

#### 7.3 Framing

Speech is not analyzed as one long block. It is cut into short overlapping frames.

Configuration:

| Setting | Value |
|---|---:|
| Sample rate | 16000 Hz |
| Frame duration | 0.025 s |
| Hop duration | 0.010 s |
| Frame length | 400 samples |
| Hop length | 160 samples |

Why framing is used:

- Speech characteristics change over time.
- Short frames let the backend measure local energy, voicing, and spectral shape.
- A 25 ms window with 10 ms hop is a common speech-processing choice.

If a signal is shorter than one frame, the backend zero-pads it so at least one frame can be analyzed.

### 8. Feature extraction details

The backend uses four feature groups.

#### 8.1 Short-term average energy

For each frame, energy is computed as the mean squared amplitude:

```text
energy(frame) = mean(frame^2)
```

Then the backend keeps two summary statistics across all frames:

- mean energy
- standard deviation of energy

Why this helps:

- Energy tracks the speech envelope.
- Voiced speech usually has stronger energy than silence or very weak regions.
- The variance of energy tells the model whether the clip is dynamically steady or highly changing.

This contributes 2 dimensions to the final vector.

#### 8.2 Short-time zero crossing rate (STZCR / ZCR)

For each frame, the backend counts how often the signal changes sign.

Implementation idea:

```text
zcr = sign changes / (2 * frame_length)
```

The code treats zeros as `-1` before counting sign changes.

Then it keeps:

- mean ZCR
- standard deviation of ZCR

Why this helps:

- Voiced sounds tend to be smoother and have lower ZCR.
- Unvoiced fricatives and noisier consonants usually have higher ZCR.
- This gives a simple time-domain clue about excitation behavior.

This contributes 2 dimensions.

#### 8.3 Linear predictive analysis (LPA) using LPC

This is the most "vocal tract" oriented part of the feature set.

The backend first chooses active frames for LPC analysis:

- It computes the energy of every frame.
- It sets an energy floor as `max(1e-6, 35th percentile of frame energies)`.
- Only frames whose energy is at least that floor are used for LPC.

Why this helps:

- LPC is more meaningful on speech-active frames than on silence.
- Excluding low-energy frames reduces the chance that silence dominates the vocal-tract estimate.

If no frames pass the energy test, the backend falls back to using all frames.

For each active frame:

- The backend runs `librosa.lpc(frame, order=12)`.
- LPC order 12 means the vocal tract is approximated with 12 predictive coefficients.
- `librosa.lpc` returns 13 coefficients including the leading coefficient, and the backend drops the first one.

Then:

- all valid 12-coefficient LPC vectors are averaged across active frames
- invalid frames are skipped
- all-zero frames are skipped
- if every LPC attempt fails, the backend returns a zero vector of length 12

Why LPC helps:

- LPC models the spectral envelope of speech.
- That envelope is related to vocal tract shape and resonance.
- Different speakers often produce measurably different resonance patterns.

This contributes 12 dimensions.

Note on terminology:

- If your course or notes use the term **LPA** (Linear Predictive Analysis), this code implements that analysis and stores the result as averaged **LPC** coefficients.

#### 8.4 Log-mel spectral summary

The backend also computes a mel spectrogram from the whole normalized signal.

Settings:

- `n_fft = 400`
- `hop_length = 160`
- `n_mels = 20`
- `power = 2.0`

Then it:

1. converts the mel spectrogram to decibels with `librosa.power_to_db(..., ref=np.max)`
2. averages each mel band over time

Why this helps:

- Mel filters summarize spectral energy in a perceptually meaningful scale.
- Averaged log-mel energy captures how the speaker's energy is distributed across low to high frequency regions.
- This complements LPC: LPC focuses on predictive vocal-tract modeling, while mel bands give a broader spectral summary.

This contributes 20 dimensions.

### 9. Final feature vector layout

The four feature groups are concatenated in this order:

1. `energy_mean`
2. `energy_std`
3. `zcr_mean`
4. `zcr_std`
5. `lpc_1` to `lpc_12`
6. `mel_1` to `mel_20`

Total:

- `2 + 2 + 12 + 20 = 36` dimensions

The backend also saves this schema to `feature_schema.json`.

### 10. The model is trained

Once all training files have been converted into feature vectors:

- the vectors are stacked into a feature matrix of shape `(number_of_files, 36)`
- the speaker labels are collected into a label array

Before fitting the model, the backend checks:

- at least one valid training file exists
- at least two speaker folders are present

If there is only one speaker, training is rejected because classification needs at least two classes.

#### 10.1 StandardScaler

The backend fits a `StandardScaler` on the full feature matrix and transforms the features.

Why scaling matters:

- Energy, ZCR, LPC, and mel features do not naturally live on the same numeric scale.
- An SVM with an RBF kernel is distance-sensitive.
- Without scaling, large-magnitude features could dominate the decision boundary.

So the backend standardizes every feature dimension to a roughly zero-mean, unit-variance space before training.

#### 10.2 LabelEncoder

The backend fits a `LabelEncoder` on the string speaker names.

This is where internal integer classes are created.

Example:

```text
"Alice" -> 0
"Bob"   -> 1
"Carol" -> 2
```

Important:

- This mapping is internal.
- The files are not renamed.
- The saved `training_labels.npy` still contains the original string speaker names.
- The integer encoding exists so the classifier can train on numeric class IDs.

#### 10.3 SVM classifier

The classifier is:

- `SVC(kernel="rbf", probability=True)`

Why this model fits the project:

- RBF SVM can model non-linear class boundaries.
- It works well on small-to-medium sized feature sets.
- `probability=True` allows the backend to call `predict_proba(...)` later during open-set prediction.

That probability support is important because the system uses confidence-based rejection.

### 11. Training artifacts are saved

After fitting, the backend writes these files into `backend/artifacts/`:

- `model.joblib`
- `scaler.joblib`
- `encoder.joblib`
- `training_features.npy`
- `training_labels.npy`
- `training_manifest.json`
- `feature_schema.json`

The manifest includes:

- speaker count
- sample count
- feature dimension
- probability threshold percentage
- per-speaker utterance counts
- classifier description
- scaler description

### 12. The training response is returned to the frontend

If training succeeds, `/train` returns JSON containing:

- a success message
- the training manifest
- the feature dimension
- a human-readable summary of the feature pipeline

The frontend then unlocks the testing panel.

## Prediction Workflow

After training, the user can upload one audio file for identification.

### 1. The frontend sends `POST /predict`

The frontend:

- collects one `.wav`, `.mp3`, or `.flac` file
- places it in a `FormData` field named `audio`
- sends it to `/predict`

### 2. The backend checks that training already happened

Before prediction, the backend tries to load `training_manifest.json`.

If it does not exist, the request is rejected with:

- HTTP `400`
- Message: `Train the model before requesting predictions.`

### 3. The uploaded test audio is validated and stored

The backend:

- sanitizes the filename with `secure_filename(...)`
- rejects unsupported extensions
- saves the file into `backend/workspace/uploads/`

Unlike the training ZIP, the prediction audio is deleted at the end of the request in a `finally` block.

### 4. The exact same feature pipeline is applied

Prediction uses the same `extract_feature_vector(...)` function as training.

This is important because:

- the model only makes sense if train-time and test-time features are computed the same way
- the same sample rate, framing, normalization, LPC, and mel settings are reused

### 5. The backend loads all saved artifacts

For prediction, the backend loads:

- the SVM
- the scaler
- the label encoder
- the saved raw training feature matrix
- the saved string training labels

Then it:

- scales the new query vector with the fitted scaler
- scales the saved training features with the same scaler

The second step is needed for distance-based open-set checking.

### 6. The SVM produces class probabilities

The backend runs:

- `classifier.predict_proba(scaled_feature_vector)`

Then it:

- finds the highest probability
- takes the corresponding encoded class ID
- converts that class ID back to the speaker name using `LabelEncoder.inverse_transform(...)`

At this point, the backend has a best candidate speaker and a confidence score.

### 7. Open-set rejection logic

The project does not accept every top-1 prediction blindly. It uses two guards.

#### Guard 1: Probability threshold

The highest class probability must be at least:

- `0.60` or `60%`

If not, the backend rejects the sample with the message:

- `I don't recognize you.`

Why this helps:

- Low confidence usually means the sample does not resemble any known speaker strongly enough.

#### Guard 2: Acoustic distance consistency

The backend also checks whether the query is acoustically close enough to the predicted speaker's own training cluster.

How it works:

1. It selects all scaled training vectors whose string label matches the predicted speaker.
2. It measures the query's minimum distance to that class using `pairwise_distances(...)`.
3. It computes a speaker-specific radius from within-class nearest-neighbor distances.

The class radius is:

- the larger of:
  - the 95th percentile of within-class nearest-neighbor distances
  - `mean + 2 * std` of within-class nearest-neighbor distances

Why this helps:

- A sample might get the highest class probability even if it is still far from the true cluster of that class.
- The distance check reduces the chance of forcing an unfamiliar voice into the nearest known speaker.

Special case:

- If a speaker has only one training sample, the radius is infinite.
- In that case, only the probability threshold really constrains recognition for that speaker.

### 8. Final recognition decision

The backend marks the sample as recognized only if both conditions are true:

- confidence >= 60%
- query distance <= predicted speaker radius

If either test fails, the system rejects the speaker.

Possible rejection reasons:

- `low_confidence`
- `outside_speaker_profile`

### 9. Prediction response

The `/predict` route returns JSON with:

- `recognized`
- `speaker`
- `confidence`
- `threshold`
- `acoustic_distance`
- `speaker_profile_radius`
- `rejection_reason`
- `message`

Example outcomes:

- recognized speaker: `Speaker identified as Alice.`
- rejected speaker: `I don't recognize you.`

## Hidden Behaviors and Practical Notes

These are easy to miss if you only look at the UI:

- Training always wipes previous artifacts and extracted data first.
- The uploaded training ZIP also remains in `backend/workspace/uploads/` until the next training reset.
- Prediction does not wipe the model; it only deletes the uploaded test audio afterward.
- The extracted training corpus remains on disk until the next training run or a training failure resets it.
- Training errors call `reset_training_state()` again, so failed runs do not leave partial models behind.
- The app allows cross-origin requests through `CORS(app)`, which is convenient for development.
- The backend has a `/health` endpoint for simple readiness checks.
- Processing order is deterministic because training files are sorted.
- Top-level hidden directories are ignored when deciding whether to descend into a single wrapper directory.
- Very short signals are padded to one frame.
- Empty signals raise an error instead of silently producing nonsense.
- If LPC fails on every usable frame, the LPC portion becomes all zeros.

## What The Server Does Not Do

To avoid confusion, here are some things the backend does not do:

- It does not rename uploaded audio files.
- It does not tag filenames with integer IDs.
- It does not perform speaker diarization inside a long conversation.
- It does not segment one audio file into multiple speaker turns.
- It does not keep multiple trained models at once.
- It does not train asynchronously in the background.

Each training audio file is treated as one sample belonging to exactly one speaker folder.

## API Summary

### `GET /health`

Returns a simple health response:

```json
{
  "status": "ok",
  "service": "who-is-speaking-backend"
}
```

### `POST /train`

Request:

- Content type: `multipart/form-data`
- Field name: `dataset`
- Value: one `.zip` archive up to 100 MB

Success behavior:

- clears old artifacts
- extracts the dataset
- computes features
- trains the scaler, label encoder, and SVM
- saves artifacts
- returns training summary JSON

### `POST /predict`

Request:

- Content type: `multipart/form-data`
- Field name: `audio`
- Value: one `.wav`, `.mp3`, or `.flac` file

Success behavior:

- loads the saved artifacts
- extracts the same 36-dimensional feature vector
- predicts the most likely speaker
- runs open-set rejection checks
- returns a recognition or rejection JSON response

## Why These Features Work Together

The four feature groups complement each other:

- Energy captures how strong speech activity is over time.
- ZCR captures how noisy or voiced the excitation pattern is.
- LPC captures vocal-tract resonance structure.
- Log-mel energies capture broader spectral balance on a perceptual frequency scale.

Together, they provide an interpretable acoustic fingerprint without hiding everything inside a deep neural network.

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

Backend dependencies:

- Flask
- Flask-Cors
- joblib
- librosa
- numpy
- scikit-learn
- soundfile

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

By default, the Vite dev server proxies:

- `/health`
- `/train`
- `/predict`

to:

- `http://127.0.0.1:5000`

The frontend also supports `VITE_API_BASE_URL` if you want to point it at another backend URL.

## One-Screen Summary

After the user uploads a ZIP file, the backend:

1. validates the request and clears old state
2. saves and safely extracts the archive
3. discovers audio files and speaker labels from folder names
4. loads every audio clip at 16 kHz mono
5. normalizes and frames the signal
6. computes energy, ZCR, LPC, and log-mel features
7. concatenates them into a 36-dimensional vector per file
8. standardizes all vectors with `StandardScaler`
9. converts speaker names to internal integer classes with `LabelEncoder`
10. trains an RBF SVM
11. saves the model and metadata
12. uses the same pipeline at prediction time, then applies confidence and distance-based open-set rejection

That is the complete server-side workflow from ZIP upload to final speaker decision.
