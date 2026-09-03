import "./HistoryPage.css"

function HistoryPage() {
  const history = [
    {
      file: "sample_video.mp4",
      result: "FAKE",
      confidence: "91%",
      date: "Today"
    },
    {
      file: "audio_sample.wav",
      result: "REAL",
      confidence: "87%",
      date: "Yesterday"
    }
  ]

  return (
    <div className="history-page">
      <div className="history-container">

        <h1 className="history-title">
          Analysis History
        </h1>

        <p className="history-description">
          Review your previous deepfake analysis results.
        </p>

        <div className="history-list">
          {history.map((item, index) => (
            <div className="history-card" key={index}>

              <div className="history-card-header">
                <h3 className="history-file">
                  {item.file}
                </h3>

                <span className="history-status">
                  {item.result}
                </span>
              </div>

              <div className="history-details">
                <span>
                  Confidence: {item.confidence}
                </span>

                <span>
                  {item.date}
                </span>
              </div>

            </div>
          ))}
        </div>

      </div>
    </div>
  )
}

export default HistoryPage