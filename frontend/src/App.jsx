import { startTransition, useMemo, useState } from 'react'

const theoryCards = [
  {
    title: 'Short-Term Energy',
    description:
      'Tracks the speech envelope frame by frame so the classifier can separate voiced activity from silence and weak background regions.',
    accent: 'cyan',
  },
  {
    title: 'Zero Crossing Rate',
    description:
      'Measures sign changes to separate smoother voiced segments from noisier unvoiced consonants such as fricatives.',
    accent: 'magenta',
  },
  {
    title: 'LPC Order 12',
    description:
      'Models the vocal tract filter using 12 linear prediction coefficients, giving an interpretable resonance signature for each speaker.',
    accent: 'cyan',
  },
  {
    title: '20 Mel Bands',
    description:
      'Averages the first 20 log-mel filter-bank energies to capture speaker-dependent spectral balance in a perceptually meaningful scale.',
    accent: 'magenta',
  },
]

const pipelineSteps = [
  'Upload a zipped training corpus organised into speaker folders.',
  'Extract a 36-dimensional feature vector from every utterance.',
  'Normalize the features with StandardScaler and train an SVM with an rbf kernel.',
  "Reject unknown voices when the prediction confidence falls below 60 percent.",
]

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? ''

function formatFileSize(file) {
  if (!file) {
    return 'No file selected'
  }

  const sizeInMegabytes = file.size / (1024 * 1024)
  return `${file.name} • ${sizeInMegabytes.toFixed(2)} MB`
}

function formatConfidence(value) {
  if (typeof value !== 'number') {
    return '--'
  }

  return `${value.toFixed(2)}%`
}

function App() {
  const [trainingFile, setTrainingFile] = useState(null)
  const [testFile, setTestFile] = useState(null)
  const [trainingResult, setTrainingResult] = useState(null)
  const [predictionResult, setPredictionResult] = useState(null)
  const [trainingError, setTrainingError] = useState('')
  const [predictionError, setPredictionError] = useState('')
  const [isTraining, setIsTraining] = useState(false)
  const [isPredicting, setIsPredicting] = useState(false)
  const [testingUnlocked, setTestingUnlocked] = useState(false)

  const speakerSummary = useMemo(
    () => trainingResult?.training_summary?.speakers ?? [],
    [trainingResult],
  )

  async function handleTrainingSubmit(event) {
    event.preventDefault()
    if (!trainingFile) {
      setTrainingError('Choose a zipped training corpus before starting.')
      return
    }

    const formData = new FormData()
    formData.append('dataset', trainingFile)

    setIsTraining(true)
    setTrainingError('')
    setPredictionError('')
    setPredictionResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/train`, {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.message || 'Training failed.')
      }

      setTrainingResult(payload)
      startTransition(() => {
        setTestingUnlocked(true)
      })
    } catch (error) {
      setTrainingResult(null)
      setTestingUnlocked(false)
      setTrainingError(error.message)
    } finally {
      setIsTraining(false)
    }
  }

  async function handlePredictionSubmit(event) {
    event.preventDefault()
    if (!testFile) {
      setPredictionError('Choose an audio sample before testing.')
      return
    }

    const formData = new FormData()
    formData.append('audio', testFile)

    setIsPredicting(true)
    setPredictionError('')

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.message || 'Prediction failed.')
      }

      setPredictionResult(payload)
    } catch (error) {
      setPredictionResult(null)
      setPredictionError(error.message)
    } finally {
      setIsPredicting(false)
    }
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-void text-copy">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_left,_rgba(78,252,255,0.22),_transparent_32%), radial-gradient(circle_at_top_right,_rgba(155,92,255,0.18),_transparent_28%),radial-gradient(circle_at_bottom,_rgba(0,184,212,0.12),_transparent_40%)]" />
      <div className="grid-overlay pointer-events-none absolute inset-0 opacity-60" />

      <main className="relative mx-auto flex min-h-screen w-full max-w-7xl flex-col gap-8 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">
        <section className="panel-shell relative overflow-hidden">
          <div className="absolute -left-16 top-10 h-36 w-36 rounded-full bg-cyan-glow/15 blur-3xl" />
          <div className="absolute right-0 top-0 h-56 w-56 rounded-full bg-magenta-glow/10 blur-3xl" />

          <div className="relative grid gap-8 lg:grid-cols-[1.35fr_0.95fr]">
            <div className="space-y-6">
              <span className="inline-flex w-fit rounded-full border border-cyan-glow/35 bg-cyan-glow/10 px-3 py-1 text-xs font-semibold uppercase tracking-[0.28em] text-cyan-glow">
                Explainable Speaker Recognition
              </span>

              <div className="space-y-4">
                <p className="font-display text-sm uppercase tracking-[0.45em] text-muted">
                  Speech Processing Course Project
                </p>
                <h1 className="max-w-3xl font-display text-4xl leading-none text-white sm:text-5xl lg:text-6xl">
                  Who&apos;s Speaking
                </h1>
                <p className="max-w-3xl text-base leading-7 text-slate-200 sm:text-lg">
                  A full-stack speaker identification system that stays academically transparent by
                  using physically interpretable time-domain and spectral features instead of
                  end-to-end deep learning.
                </p>
              </div>

              <div className="grid gap-4 sm:grid-cols-3">
                <div className="metric-chip">
                  <span className="metric-label">Feature Vector</span>
                  <strong className="metric-value">36 dimensions</strong>
                </div>
                <div className="metric-chip">
                  <span className="metric-label">Open Set</span>
                  <strong className="metric-value">60% threshold</strong>
                </div>
                <div className="metric-chip">
                  <span className="metric-label">Classifier</span>
                  <strong className="metric-value">SVM RBF</strong>
                </div>
              </div>
            </div>

            <div className="relative overflow-hidden rounded-[28px] border border-white/10 bg-slate-950/65 p-6 shadow-[0_0_40px_rgba(0,0,0,0.35)]">
              <div className="mb-5 flex items-center justify-between">
                <div>
                  <p className="font-display text-lg uppercase tracking-[0.28em] text-cyan-glow">
                    Signal Path
                  </p>
                  <p className="mt-2 text-sm text-muted">
                    Each utterance is framed, summarized, normalized, and classified.
                  </p>
                </div>
                <div className="signal-loader" aria-hidden="true">
                  <span />
                  <span />
                  <span />
                  <span />
                  <span />
                </div>
              </div>

              <ol className="space-y-3">
                {pipelineSteps.map((step, index) => (
                  <li
                    key={step}
                    className="flex items-start gap-3 rounded-2xl border border-white/6 bg-white/3 px-4 py-3"
                  >
                    <span className="mt-0.5 inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-cyan-glow/15 text-sm font-semibold text-cyan-glow">
                      {index + 1}
                    </span>
                    <span className="text-sm leading-6 text-slate-200">{step}</span>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="panel-shell">
            <div className="mb-6 flex items-end justify-between gap-4">
              <div>
                <p className="section-tag">Feature Theory</p>
                <h2 className="section-title">Physics-Based Acoustic Fingerprints</h2>
              </div>
              <p className="max-w-sm text-right text-sm leading-6 text-muted">
                The model learns speaker identity from envelope, excitation pattern, vocal tract
                resonance, and mel-scale spectral distribution.
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              {theoryCards.map((card) => (
                <article
                  key={card.title}
                  className={`feature-card ${card.accent === 'cyan' ? 'feature-card-cyan' : 'feature-card-magenta'}`}
                >
                  <p className="feature-title">{card.title}</p>
                  <p className="text-sm leading-6 text-slate-200">{card.description}</p>
                </article>
              ))}
            </div>
          </div>

          <div className="panel-shell">
            <p className="section-tag">Decision Logic</p>
            <h2 className="section-title">Open-Set Identification</h2>
            <div className="mt-5 space-y-4 text-sm leading-6 text-slate-200">
              <p>
                The backend uses <span className="font-semibold text-white">predict_proba</span> to
                estimate confidence for each known speaker class after SVM classification.
              </p>
              <p>
                If the highest confidence is below{' '}
                <span className="font-semibold text-cyan-glow">60 percent</span>, the system
                rejects the input and returns the message{' '}
                <span className="font-semibold text-white">I don&apos;t recognize you.</span>
              </p>
              <p>
                A second open-set guard compares the sample against the predicted speaker&apos;s
                nearest training neighbors in standardized feature space so unfamiliar voices are
                less likely to be forced into a known class.
              </p>
              <div className="rounded-2xl border border-magenta-glow/20 bg-magenta-glow/8 p-4">
                <p className="text-xs uppercase tracking-[0.28em] text-magenta-glow">
                  Current Modalities
                </p>
                <p className="mt-2 text-sm text-slate-200">
                  File-based training and testing are enabled now. Live microphone capture can be
                  added later without changing the feature pipeline.
                </p>
              </div>
            </div>
          </div>
        </section>

        <section className="grid gap-6 xl:grid-cols-[1.06fr_0.94fr]">
          <form className="panel-shell space-y-6" onSubmit={handleTrainingSubmit}>
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="section-tag">Step 1</p>
                <h2 className="section-title">Train the Model</h2>
              </div>
              <span className="rounded-full border border-white/10 px-3 py-1 text-xs uppercase tracking-[0.25em] text-muted">
                Zip up to 100 MB
              </span>
            </div>

            <p className="text-sm leading-6 text-slate-200">
              Upload a single archive where each folder name is a speaker label. Supported audio
              files are <span className="font-semibold text-white">.wav</span>,{' '}
              <span className="font-semibold text-white">.mp3</span>, and{' '}
              <span className="font-semibold text-white">.flac</span>.
            </p>

            <label className="upload-shell">
              <span className="upload-title">Training Corpus</span>
              <span className="upload-caption">{formatFileSize(trainingFile)}</span>
              <input
                type="file"
                accept=".zip"
                className="sr-only"
                onChange={(event) => {
                  setTrainingFile(event.target.files?.[0] ?? null)
                  setTrainingError('')
                }}
              />
              <span className="upload-button">Choose .zip</span>
            </label>

            <button type="submit" className="action-button" disabled={isTraining}>
              {isTraining ? 'Extracting Features and Fitting SVM...' : 'Train Model'}
            </button>

            {isTraining ? (
              <div className="rounded-2xl border border-cyan-glow/20 bg-cyan-glow/8 p-4">
                <p className="text-sm font-semibold text-cyan-glow">Processing acoustic evidence</p>
                <p className="mt-2 text-sm text-slate-200">
                  Computing frame energy, ZCR, LPC resonances, log-mel summaries, scaling, and
                  probability calibration.
                </p>
                <div className="mt-4 flex gap-2" aria-hidden="true">
                  <span className="scan-pill scan-pill-1" />
                  <span className="scan-pill scan-pill-2" />
                  <span className="scan-pill scan-pill-3" />
                </div>
              </div>
            ) : null}

            {trainingError ? (
              <div className="status-card border-danger/35 bg-danger/10 text-rose-100" role="alert">
                {trainingError}
              </div>
            ) : null}

            {trainingResult ? (
              <div className="space-y-4 rounded-[26px] border border-success/25 bg-success/8 p-5">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <p className="text-xs uppercase tracking-[0.28em] text-success">
                      Training Complete
                    </p>
                    <p className="mt-2 text-lg font-semibold text-white">
                      Now I recognize you.
                    </p>
                  </div>
                  <div className="rounded-full border border-success/25 px-3 py-1 text-xs uppercase tracking-[0.25em] text-success">
                    Ready for testing
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="mini-stat">
                    <span className="mini-stat-label">Speakers</span>
                    <strong className="mini-stat-value">
                      {trainingResult.training_summary.speaker_count}
                    </strong>
                  </div>
                  <div className="mini-stat">
                    <span className="mini-stat-label">Utterances</span>
                    <strong className="mini-stat-value">
                      {trainingResult.training_summary.sample_count}
                    </strong>
                  </div>
                  <div className="mini-stat">
                    <span className="mini-stat-label">Vector Size</span>
                    <strong className="mini-stat-value">{trainingResult.feature_dimension}</strong>
                  </div>
                </div>

                <div className="grid gap-3 md:grid-cols-2">
                  {speakerSummary.map((speaker) => (
                    <div key={speaker.name} className="speaker-pill">
                      <span className="font-semibold text-white">{speaker.name}</span>
                      <span className="text-sm text-muted">{speaker.utterances} files</span>
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </form>

          <form
            className={`panel-shell space-y-6 transition duration-500 ${
              testingUnlocked
                ? 'border-cyan-glow/20 shadow-[0_0_42px_rgba(78,252,255,0.08)]'
                : 'border-white/8 opacity-70'
            }`}
            onSubmit={handlePredictionSubmit}
          >
            <div className="flex flex-wrap items-start justify-between gap-4">
              <div>
                <p className="section-tag">Step 2</p>
                <h2 className="section-title">Test a Voice Sample</h2>
              </div>
              <span className="rounded-full border border-white/10 px-3 py-1 text-xs uppercase tracking-[0.25em] text-muted">
                Open-set prediction
              </span>
            </div>

            {!testingUnlocked ? (
              <div className="rounded-[26px] border border-white/8 bg-white/3 p-5">
                <p className="text-lg font-semibold text-white">Training must finish first.</p>
                <p className="mt-2 text-sm leading-6 text-muted">
                  Once the SVM is trained successfully, this panel unlocks and the system can judge
                  whether a test sample belongs to a known speaker or should be rejected.
                </p>
              </div>
            ) : (
              <>
                <p className="text-sm leading-6 text-slate-200">
                  Upload a single audio sample and the backend will either identify the speaker or
                  reject it if the highest class probability falls below the 60 percent threshold.
                </p>

                <label className="upload-shell">
                  <span className="upload-title">Test Utterance</span>
                  <span className="upload-caption">{formatFileSize(testFile)}</span>
                  <input
                    type="file"
                    accept=".wav,.mp3,.flac"
                    className="sr-only"
                    onChange={(event) => {
                      setTestFile(event.target.files?.[0] ?? null)
                      setPredictionError('')
                    }}
                  />
                  <span className="upload-button">Choose audio</span>
                </label>

                <button type="submit" className="action-button" disabled={isPredicting}>
                  {isPredicting ? 'Comparing Against Known Speakers...' : 'Identify Speaker'}
                </button>

                {predictionError ? (
                  <div className="status-card border-danger/35 bg-danger/10 text-rose-100" role="alert">
                    {predictionError}
                  </div>
                ) : null}

                {predictionResult ? (
                  <div
                    className={`rounded-[26px] border p-5 ${
                      predictionResult.recognized
                        ? 'border-cyan-glow/25 bg-cyan-glow/8'
                        : 'border-warning/30 bg-warning/10'
                    }`}
                  >
                    <p
                      className={`text-xs uppercase tracking-[0.28em] ${
                        predictionResult.recognized ? 'text-cyan-glow' : 'text-warning'
                      }`}
                    >
                      {predictionResult.recognized ? 'Speaker Match' : 'Unknown Speaker'}
                    </p>

                    <p className="mt-3 text-2xl font-semibold text-white">
                      {predictionResult.recognized
                        ? predictionResult.speaker
                        : "I don't recognize you."}
                    </p>

                    <div className="mt-5 grid gap-3 sm:grid-cols-2">
                      <div className="mini-stat">
                        <span className="mini-stat-label">Confidence</span>
                        <strong className="mini-stat-value">
                          {formatConfidence(predictionResult.confidence)}
                        </strong>
                      </div>
                      <div className="mini-stat">
                        <span className="mini-stat-label">Threshold</span>
                        <strong className="mini-stat-value">
                          {formatConfidence(predictionResult.threshold)}
                        </strong>
                      </div>
                    </div>

                    <p className="mt-4 text-sm leading-6 text-slate-200">
                      {predictionResult.message}
                    </p>
                  </div>
                ) : (
                  <div className="rounded-[26px] border border-white/8 bg-white/3 p-5">
                    <p className="text-sm leading-6 text-muted">
                      The decision card will appear here after feature extraction and SVM
                      inference.
                    </p>
                  </div>
                )}
              </>
            )}
          </form>
        </section>
      </main>
    </div>
  )
}

export default App
