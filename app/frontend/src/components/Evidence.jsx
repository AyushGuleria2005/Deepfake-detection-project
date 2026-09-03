import "./Evidence.css"

function Evidence() {
  return (
    <div className="evidence">

      <h2 className="evidence-title">
        Detection Evidence
      </h2>

      <div className="evidence-grid">

        <div className="evidence-card">
          <h3>Visual Evidence</h3>
          <p>
            Face manipulation indicators detected in the video.
          </p>
        </div>

        <div className="evidence-card">
          <h3>Audio Evidence</h3>
          <p>
            Synthetic audio characteristics detected in the audio.
          </p>
        </div>

        <div className="evidence-card">
          <h3>Cross-Modal Evidence</h3>
          <p>
            Video and audio signals show potential inconsistency.
          </p>
        </div>

      </div>

      <div className="evidence-explanation">
        <h3>Explanation</h3>
        <p>
          Multiple AI signals contributed to the final detection
          decision.
        </p>
      </div>

    </div>
  )
}

export default Evidence