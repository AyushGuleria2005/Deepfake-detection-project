import Upload from "../components/Upload"
import "./Analyze.css"

function Analyze() {
  return (
    <div className="analyze-page">
      <div className="analyze-header">
        <div className="page-tag">Inspection Workspace</div>
        <h1 className="analyze-title">Neural Media Analysis</h1>
        <p className="analyze-description">
          Upload media to evaluate facial inconsistencies, boundary artifacts, and temporal frame sequences using deep convolutional and recurrent models.
        </p>
      </div>

      <div className="analyze-workspace">
        <Upload />
      </div>

      <div className="model-specs-grid">
        <div className="spec-card">
          <div className="spec-icon">🔬</div>
          <div className="spec-body">
            <h4>ResNet18 Fine-Tuned Backbone</h4>
            <p>Trained with face crop normalization, CutMix, and label smoothing for spatial artifact detection.</p>
          </div>
        </div>
        <div className="spec-card">
          <div className="spec-icon">⏱️</div>
          <div className="spec-body">
            <h4>Bidirectional LSTM + Attention</h4>
            <p>Inspects inter-frame micro-fluctuations and temporal coherence across sequential video frames.</p>
          </div>
        </div>
        <div className="spec-card">
          <div className="spec-icon">🛡️</div>
          <div className="spec-body">
            <h4>Explainable Forensic Evidence</h4>
            <p>Produces granular confidence metrics and decision indicators for verifiable analysis.</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analyze
