import { useState, useEffect } from "react"
import { Link } from "react-router-dom"
import "./HistoryPage.css"

function HistoryPage() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    try {
      const stored = localStorage.getItem("df_history")
      if (stored) {
        setHistory(JSON.parse(stored))
      } else {
        // Initial sample seed with updated visual/video items (no audio)
        const defaultHistory = [
          {
            file: "interview_sequence_01.mp4",
            result: "FAKE",
            confidence: "94.2%",
            date: "Today, 14:20",
            type: "Video"
          },
          {
            file: "press_conference_crop.png",
            result: "REAL",
            confidence: "89.6%",
            date: "Yesterday",
            type: "Image"
          },
          {
            file: "speech_clip_test.mp4",
            result: "FAKE",
            confidence: "91.8%",
            date: "Nov 12",
            type: "Video"
          }
        ]
        setHistory(defaultHistory)
        localStorage.setItem("df_history", JSON.stringify(defaultHistory))
      }
    } catch (e) {
      console.warn("Error accessing history storage", e)
    }
  }, [])

  const clearHistory = () => {
    localStorage.removeItem("df_history")
    setHistory([])
  }

  return (
    <div className="history-page">
      <div className="history-header">
        <div className="header-text">
          <div className="page-tag">Audit Logs</div>
          <h1 className="history-title">Analysis History</h1>
          <p className="history-description">
            Audit trail of submitted image and video sequences evaluated by the deepfake forensic engine.
          </p>
        </div>

        {history.length > 0 && (
          <button className="clear-btn" onClick={clearHistory}>
            Clear History
          </button>
        )}
      </div>

      {history.length === 0 ? (
        <div className="empty-history-card">
          <div className="empty-icon">📂</div>
          <h3>No Analysis History</h3>
          <p>You haven't inspected any video or image files yet.</p>
          <Link to="/analyze" className="empty-cta">
            Inspect Media Now
          </Link>
        </div>
      ) : (
        <div className="history-table-wrapper">
          <table className="history-table">
            <thead>
              <tr>
                <th>Target Media</th>
                <th>Type</th>
                <th>Classification</th>
                <th>Confidence</th>
                <th>Timestamp</th>
                <th className="text-right">Action</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item, index) => {
                const isFake = item.result.toUpperCase() === "FAKE"
                return (
                  <tr key={index}>
                    <td className="file-cell font-mono">
                      <span className="file-icon-mini">{item.type === "Image" ? "🖼️" : "🎬"}</span>
                      {item.file}
                    </td>
                    <td>
                      <span className="type-badge font-mono">{item.type || "Video"}</span>
                    </td>
                    <td>
                      <span className={`status-badge font-mono ${isFake ? "badge-fake" : "badge-real"}`}>
                        {item.result}
                      </span>
                    </td>
                    <td className="font-mono text-secondary">
                      {item.confidence}
                    </td>
                    <td className="font-mono text-muted">
                      {item.date}
                    </td>
                    <td className="text-right">
                      <Link to="/result" className="inspect-link">
                        Review →
                      </Link>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default HistoryPage
