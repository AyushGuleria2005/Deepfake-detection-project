import Upload from "../components/Upload"
import "./Analyze.css"

function Analyze() {
  return (
    <div className="analyze">
      <div className="analyze-container">

        <h1 className="analyze-title">
          Deepfake Analysis
        </h1>

        <p className="analyze-description">
          Upload a video or audio file for multimodal AI analysis.
        </p>

        <Upload />

      </div>
    </div>
  )
}

export default Analyze