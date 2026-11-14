import { useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import axios from "axios"
import "./Upload.css"

function Upload() {
  const fileInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [error, setError] = useState("")

  const navigate = useNavigate()

  const handleFile = (selectedFile) => {
    if (!selectedFile) return
    // Support image and video formats
    const validTypes = [
      "image/jpeg", "image/png", "image/webp",
      "video/mp4", "video/quicktime", "video/x-msvideo", "video/webm"
    ]
    const ext = selectedFile.name.split(".").pop().toLowerCase()
    const validExts = ["mp4", "avi", "mov", "webm", "jpg", "jpeg", "png", "webp"]

    if (!validTypes.includes(selectedFile.type) && !validExts.includes(ext)) {
      setError("Please select a supported video or image file (.mp4, .mov, .avi, .jpg, .png).")
      return
    }

    setFile(selectedFile)
    setError("")
  }

  const handleDrop = (event) => {
    event.preventDefault()
    setIsDragging(false)
    const droppedFile = event.dataTransfer.files[0]
    handleFile(droppedFile)
  }

  const handleAnalyze = async (event) => {
    event.stopPropagation()
    if (!file) return

    setUploading(true)
    setError("")
    setUploadProgress(15)

    const formData = new FormData()
    formData.append("file", file)

    const isVideo = file.type.startsWith("video") || ["mp4", "mov", "avi", "webm"].includes(file.name.split(".").pop().toLowerCase())
    const endpoint = isVideo ? "http://localhost:8000/analyze" : "http://localhost:8000/predict-image"

    try {
      const progressTimer = setInterval(() => {
        setUploadProgress((prev) => (prev < 90 ? prev + 12 : prev))
      }, 350)

      const response = await axios.post(endpoint, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })

      clearInterval(progressTimer)
      setUploadProgress(100)

      if (response.data && response.data.success === false) {
        throw new Error(response.data.error || "Analysis failed")
      }

      // Store in session storage for persistence and pass in state
      sessionStorage.setItem("last_analysis", JSON.stringify({
        data: response.data,
        fileName: file.name,
        fileSize: (file.size / (1024 * 1024)).toFixed(2) + " MB",
        isVideo,
        timestamp: new Date().toISOString()
      }))

      // Also record in history list in localStorage
      try {
        const existingHistory = JSON.parse(localStorage.getItem("df_history") || "[]")
        const resultLabel = response.data.prediction || (response.data.label ? response.data.label.toUpperCase() : "ANALYZED")
        const confVal = response.data.confidence 
          ? Math.round(response.data.confidence * 100) + "%" 
          : (response.data.probability ? Math.round(response.data.probability * 100) + "%" : "85%")

        const historyItem = {
          file: file.name,
          result: resultLabel,
          confidence: confVal,
          date: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
          type: isVideo ? "Video" : "Image"
        }
        localStorage.setItem("df_history", JSON.stringify([historyItem, ...existingHistory.slice(0, 19)]))
      } catch (e) {
        console.warn("Could not save to history storage", e)
      }

      setTimeout(() => {
        navigate("/result", {
          state: {
            result: response.data,
            fileName: file.name,
            fileSize: (file.size / (1024 * 1024)).toFixed(2) + " MB",
            isVideo
          }
        })
      }, 400)
    } catch (err) {
      console.error(err)
      setError(err.response?.data?.detail || err.message || "Failed to connect to backend engine at localhost:8000")
    } finally {
      setUploading(false)
    }
  }

  const isVideo = file ? (file.type.startsWith("video") || ["mp4", "mov", "avi", "webm"].includes(file.name.split(".").pop().toLowerCase())) : false

  return (
    <div className="upload-container">
      <div
        className={`drop-zone ${isDragging ? "dragging" : ""} ${file ? "has-file" : ""}`}
        onDragOver={(event) => {
          event.preventDefault()
          setIsDragging(true)
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => !file && fileInputRef.current.click()}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/x-msvideo,image/jpeg,image/png,image/webp"
          hidden
          onChange={(event) => handleFile(event.target.files[0])}
        />

        {!file ? (
          <div className="upload-empty">
            <div className="upload-icon-wrapper">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>

            <div className="upload-texts">
              <p className="upload-main-text">
                <span className="upload-highlight">Click to upload</span> or drag and drop
              </p>
              <p className="upload-sub-text">
                Video (MP4, MOV, AVI) or Image (JPG, PNG) up to 200MB
              </p>
            </div>

            <div className="upload-badge-row">
              <span className="format-tag">ResNet18 Visual</span>
              <span className="format-tag">Temporal BiLSTM</span>
              <span className="format-tag">Haar Cascade</span>
            </div>
          </div>
        ) : (
          <div className="upload-selected">
            <div className="file-card">
              <div className="file-avatar">
                {isVideo ? "🎬" : "🖼️"}
              </div>
              <div className="file-info">
                <p className="file-name">{file.name}</p>
                <p className="file-meta">
                  {(file.size / (1024 * 1024)).toFixed(2)} MB • {isVideo ? "Video Sequence" : "Still Image"}
                </p>
              </div>
              <button
                type="button"
                className="remove-btn"
                onClick={(e) => {
                  e.stopPropagation()
                  setFile(null)
                  setError("")
                }}
                disabled={uploading}
              >
                ✕
              </button>
            </div>

            {uploading && (
              <div className="progress-section">
                <div className="progress-labels">
                  <span>Executing Neural Inference</span>
                  <span>{uploadProgress}%</span>
                </div>
                <div className="progress-bar-track">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${uploadProgress}%` }}
                  ></div>
                </div>
              </div>
            )}

            <div className="action-row">
              <button
                type="button"
                className="btn-secondary"
                onClick={(e) => {
                  e.stopPropagation()
                  fileInputRef.current.click()
                }}
                disabled={uploading}
              >
                Select Different File
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={handleAnalyze}
                disabled={uploading}
              >
                {uploading ? (
                  <>
                    <span className="spinner"></span>
                    Analyzing Media...
                  </>
                ) : (
                  <>
                    Run DeepFake Inspection
                    <span className="btn-arrow">→</span>
                  </>
                )}
              </button>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="error-banner">
          <span className="error-icon">⚠</span>
          <span className="error-text">{error}</span>
        </div>
      )}
    </div>
  )
}

export default Upload
