import { useLocation, Link, useNavigate } from "react-router-dom"
import Evidence from "../components/Evidence"
import "./AnalysisResult.css"

function AnalysisResult() {
  const location = useLocation()
  const navigate = useNavigate()

  // Retrieve state or check sessionStorage fallback
  let payload = location.state
  if (!payload) {
    try {
      const stored = sessionStorage.getItem("last_analysis")
      if (stored) payload = JSON.parse(stored)
    } catch (e) {
      console.warn("No stored analysis", e)
    }
  }

  // Fallback mock if directly navigated
  const data = payload?.result || payload?.data || {
    prediction: "REAL",
    confidence: 0.884,
    visual: {
      prediction: "REAL",
      confidence: 0.892,
      fake_probability: 0.108,
      real_probability: 0.892,
      processed_frames: 12
    },
    temporal_lstm: {
      prediction: "REAL",
      confidence: 0.876,
      fake_probability: 0.124,
      real_probability: 0.876
    },
    bilstm_attention: {
      prediction: "REAL",
      confidence: 0.885,
      fake_probability: 0.115,
      real_probability: 0.885
    },
    evidence: {
      type: "Spatial-Temporal Alignment",
      explanation: "Facial micro-landmarks and temporal frame consistency across 12 frames confirm biological authenticity without synthetic warping."
    },
    video: {
      total_frames: 48,
      fps: 30.0,
      duration_seconds: 1.6,
      processed_frames: 12
    },
    device: "cuda:0"
  }

  const fileName = payload?.fileName || "sample_inspection.mp4"
  const fileSize = payload?.fileSize || "4.12 MB"
  const isVideo = payload?.isVideo !== undefined ? payload.isVideo : true

  const isFake = (data.prediction || data.label || "").toUpperCase() === "FAKE"
  const confidencePercent = Math.round(
    (data.confidence !== undefined ? data.confidence : (data.probability !== undefined ? data.probability : 0.88)) * 100
  )

  return (
    <div className="result-page">
      {/* Top Banner Navigation */}
      <div className="result-nav-bar">
        <Link to="/analyze" className="back-link">
          <span className="arrow">←</span> Analyze another file
        </Link>
        <span className="result-time font-mono">Inference completed • GPU accelerated</span>
      </div>

      {/* Main Verdict Card */}
      <div className={`verdict-card ${isFake ? "verdict-fake" : "verdict-real"}`}>
        <div className="verdict-header">
          <div className="verdict-tag font-mono">
            Classification Verdict
          </div>
          <div className="device-tag font-mono">
            {data.device || "NVIDIA RTX 3050"}
          </div>
        </div>

        <div className="verdict-main">
          <div className="verdict-left">
            <h1 className="verdict-label">
              {isFake ? "DEEPFAKE DETECTED" : "AUTHENTIC MEDIA"}
            </h1>
            <p className="verdict-description">
              {isFake
                ? "Neural inspection detected synthetic facial blending and temporal sequence anomalies."
                : "No synthetic manipulation indicators or unnatural frame transitions detected."}
            </p>
          </div>

          <div className="verdict-score-box">
            <span className="score-number font-mono">{confidencePercent}%</span>
            <span className="score-label">Confidence Score</span>
          </div>
        </div>

        {/* Breakdown bar */}
        <div className="verdict-bar-wrapper">
          <div className="bar-labels font-mono">
            <span>Fake Probability ({((data.visual?.fake_probability ?? (isFake ? 0.91 : 0.09)) * 100).toFixed(1)}%)</span>
            <span>Real Probability ({((data.visual?.real_probability ?? (isFake ? 0.09 : 0.91)) * 100).toFixed(1)}%)</span>
          </div>
          <div className="bar-track">
            <div
              className={`bar-fill ${isFake ? "bar-fake" : "bar-real"}`}
              style={{
                width: isFake ? `${confidencePercent}%` : `${100 - confidencePercent}%`,
              }}
            ></div>
          </div>
        </div>
      </div>

      {/* File & Meta Information */}
      <div className="meta-card">
        <div className="meta-item">
          <span className="meta-key">Analyzed Target</span>
          <span className="meta-val font-mono">{fileName}</span>
        </div>
        <div className="meta-item">
          <span className="meta-key">File Size</span>
          <span className="meta-val font-mono">{fileSize}</span>
        </div>
        <div className="meta-item">
          <span className="meta-key">Detection Models</span>
          <span className="meta-val">ResNet18 + BiLSTM</span>
        </div>
        {data.video && (
          <div className="meta-item">
            <span className="meta-key">Frames Evaluated</span>
            <span className="meta-val font-mono">{data.video.processed_frames || 5} frames ({data.video.fps || 25} FPS)</span>
          </div>
        )}
      </div>

      {/* Detailed Evidence Breakdown */}
      <div className="result-evidence-section">
        <Evidence
          evidence={data.evidence}
          visual={data.visual}
          temporal={data.temporal_lstm}
          bilstm={data.bilstm_attention}
          isVideo={isVideo}
        />
      </div>

      {/* Bottom Action Footer */}
      <div className="result-footer-actions">
        <button
          className="btn-action-outline"
          onClick={() => navigate("/history")}
        >
          View In Audit History
        </button>
        <button
          className="btn-action-primary"
          onClick={() => navigate("/analyze")}
        >
          Inspect New Sample
        </button>
      </div>
    </div>
  )
}

export default AnalysisResult
