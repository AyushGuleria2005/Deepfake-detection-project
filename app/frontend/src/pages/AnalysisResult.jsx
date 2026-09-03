import Evidence from "../components/Evidence"
import "./AnalysisResult.css"

function AnalysisResult() {
  return (
    <div className="result-page">
      <div className="result-container">

        <h1 className="result-title">
          Analysis Result
        </h1>

        <div className="result-summary">
          <p className="result-status">FAKE</p>

          <p className="result-confidence">
            Confidence: 91%
          </p>
        </div>

        <div className="result-evidence">
          <Evidence />
        </div>

      </div>
    </div>
  )
}

export default AnalysisResult