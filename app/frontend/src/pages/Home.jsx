import { Link } from "react-router-dom"

function Home() {
  return (
    <div style={{ textAlign: "center", padding: "60px 20px" }}>
      <h1 style={{ fontSize: "2.5rem", marginBottom: "16px" }}>
        Deepfake Detection System
      </h1>
      <p style={{ color: "var(--text-secondary)", marginBottom: "32px", maxWidth: "540px", margin: "0 auto 32px" }}>
        Inspect facial anomalies, spatial frequency textures, and temporal sequence continuity using fine-tuned ResNet18 and BiLSTM models.
      </p>
      <Link to="/analyze">
        <button className="cta-primary">Launch Detection Scanner →</button>
      </Link>
    </div>
  )
}

export default Home
