import { useRef, useState } from "react"
import { useNavigate } from "react-router-dom"
import axios from "axios"
import "./Upload.css"

function Upload() {
  const fileInputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState("")

  const navigate = useNavigate()

  const handleFile = (selectedFile) => {
    if (!selectedFile) return

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

    const formData = new FormData()
    formData.append("media", file)

    try {
      const response = await axios.post(
        "http://localhost:5000/api/upload",
        formData
      )

      console.log("Upload response:", response.data)

      navigate("/result")
    } catch (err) {
      console.error(err)
      setError("Upload failed. Please try again.")
    } finally {
      setUploading(false)
    }
  }

  return (
    <div
      className={`drop-zone ${isDragging ? "dragging" : ""}`}
      onDragOver={(event) => {
        event.preventDefault()
        setIsDragging(true)
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => fileInputRef.current.click()}
    >
      <input
        ref={fileInputRef}
        type="file"
        accept="video/*,audio/*"
        hidden
        onChange={(event) => handleFile(event.target.files[0])}
      />

      {!file ? (
        <>
          <div className="upload-circle">↑</div>

          <h2>Drop your media here</h2>

          <p className="drop-description">
            Drag and drop your video or audio file
          </p>

          <span className="drop-or">or</span>

          <button
            className="browse-button"
            onClick={(event) => {
              event.stopPropagation()
              fileInputRef.current.click()
            }}
          >
            Browse Files
          </button>

          <p className="supported-files">
            MP4 · MOV · AVI · WAV · MP3
          </p>
        </>
      ) : (
        <div className="file-selected">
          <div className="file-icon">✓</div>

          <h2>File Ready</h2>

          <p>{file.name}</p>

          <div className="file-actions">
            <button
              className="change-button"
              onClick={(event) => {
                event.stopPropagation()
                fileInputRef.current.click()
              }}
            >
              Change File
            </button>

            <button
              className="analyze-file-button"
              onClick={handleAnalyze}
              disabled={uploading}
            >
              {uploading ? "Uploading..." : "Analyze Media"}
            </button>
          </div>

          {error && (
            <p className="upload-error">
              {error}
            </p>
          )}
        </div>
      )}
    </div>
  )
}

export default Upload