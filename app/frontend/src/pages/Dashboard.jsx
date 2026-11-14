import { Link } from "react-router-dom"
import "./Dashboard.css"

function Dashboard() {
  const stats = [
    { label: "Image ROC-AUC", value: "80.05%", badge: "+2.73% Fine-tuned", mono: true },
    { label: "Balanced Accuracy", value: "72.41%", badge: "High Recall", mono: true },
    { label: "Active Pipeline", value: "Visual + Temporal", badge: "Live GPU", mono: false },
    { label: "Inference Latency", value: "~18ms", badge: "RTX 3050 CUDA", mono: true }
  ]

  const capabilities = [
    {
      title: "Facial Boundary Artifacts",
      desc: "Detects blending seams, unnatural skin frequency textures, and warping artifacts across isolated face crops.",
      tag: "Spatial Analysis"
    },
    {
      title: "Temporal LSTM Coherence",
      desc: "Evaluates frame-to-frame feature persistence to expose unnatural micro-blinks, twitching, or jitter.",
      tag: "Sequential RNN"
    },
    {
      title: "BiLSTM Attention Engine",
      desc: "Weights critical transition frames using self-attention mechanisms to spot transient deepfake flickers.",
      tag: "Attention Weights"
    },
    {
      title: "Interpretable Forensic Evidence",
      desc: "Delivers breakdown metrics, individual frame inspection data, and confidence scores directly in the UI.",
      tag: "Explainable AI"
    }
  ]

  return (
    <div className="dashboard-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-badge">
          <span className="hero-badge-dot"></span>
          <span>Next-Generation DeepFake Detection</span>
        </div>

        <h1 className="hero-title">
          Verify digital authenticity with forensic precision.
        </h1>

        <p className="hero-subtitle">
          An enterprise-grade deepfake detection suite powered by fine-tuned ResNet18 convolutional backbones and bidirectional temporal attention models.
        </p>

        <div className="hero-cta-group">
          <Link to="/analyze" className="cta-primary">
            Analyze Media Now
            <span className="arrow">→</span>
          </Link>
          <Link to="/history" className="cta-secondary">
            View Past Results
          </Link>
          <Link to="/learning" className="cta-secondary">
            Take Forensic Quiz
          </Link>
        </div>
      </section>

      {/* Metrics Row */}
      <section className="stats-grid">
        {stats.map((stat, i) => (
          <div key={i} className="stat-card">
            <div className="stat-header">
              <span className="stat-label">{stat.label}</span>
              <span className="stat-badge">{stat.badge}</span>
            </div>
            <div className={`stat-value ${stat.mono ? "font-mono" : ""}`}>
              {stat.value}
            </div>
          </div>
        ))}
      </section>

      {/* Capabilities Section */}
      <section className="capabilities-section">
        <div className="section-head">
          <h2 className="section-title">Core Detection Framework</h2>
          <p className="section-desc">Multi-layered architectural analysis without reliance on fragile heuristics.</p>
        </div>

        <div className="capabilities-grid">
          {capabilities.map((c, i) => (
            <div key={i} className="cap-card">
              <div className="cap-tag">{c.tag}</div>
              <h3 className="cap-title">{c.title}</h3>
              <p className="cap-desc">{c.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Architecture Overview */}
      <section className="arch-card">
        <div className="arch-inner">
          <div className="arch-text">
            <span className="arch-kicker">Unified Pipeline</span>
            <h3>Dual Spatial & Temporal Analysis</h3>
            <p>
              Uploaded video sequences undergo facial extraction via automated Haar detectors, passing 512-dimensional embedding representations through temporal recurrent layers to flag AI synthetic generation.
            </p>
          </div>
          <div className="arch-actions">
            <Link to="/analyze" className="btn-arch">
              Launch Scanner
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Dashboard
