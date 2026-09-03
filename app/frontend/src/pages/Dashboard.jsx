import { Link } from "react-router-dom"
import "./Dashboard.css"

function Dashboard() {
  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1 className="dashboard-title">
          AI Deepfake Detection
        </h1>

        <p className="dashboard-description">
          Detect, understand, and learn about manipulated media
          using multimodal AI.
        </p>

        <div className="dashboard-actions">
          <Link to="/analyze">
            <button className="dashboard-button">
              Analyze Media
            </button>
          </Link>

          <Link to="/history">
            <button className="dashboard-button">
              View History
            </button>
          </Link>

          <Link to="/learning">
            <button className="dashboard-button">
              Start Learning
            </button>
          </Link>
        </div>
      </div>

      <div className="dashboard-features">
        <h2>System Capabilities</h2>

        <ul>
          <li>Visual deepfake analysis</li>
          <li>Audio deepfake analysis</li>
          <li>Cross-modal evidence</li>
          <li>Explainable detection results</li>
          <li>Adaptive learning and quiz</li>
        </ul>
      </div>
    </div>
  )
}

export default Dashboard