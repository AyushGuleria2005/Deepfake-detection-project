const express = require("express")
const cors = require("cors")
const axios = require("axios")
const multer = require("multer")
const FormData = require("form-data")
const path = require("path")
const fs = require("fs")

const app = express()

app.use(cors())
app.use(express.json())

const uploadDir = path.join(__dirname, "uploads")

if (!fs.existsSync(uploadDir)) {
  fs.mkdirSync(uploadDir)
}

const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, uploadDir)
  },

  filename: (req, file, cb) => {
    const uniqueName =
      Date.now() + "-" + file.originalname

    cb(null, uniqueName)
  }
})

const upload = multer({ storage })

app.get("/", (req, res) => {
  res.json({
    status: "online",
    message: "Node.js backend is running"
  })
})

app.get("/api/health", async (req, res) => {
  try {
    const response = await axios.get(
      "http://127.0.0.1:8000/health"
    )

    res.json({
      node: "healthy",
      fastapi: response.data
    })
  } catch (error) {
    res.status(500).json({
      node: "healthy",
      fastapi: "unavailable"
    })
  }
})

app.post(
  "/api/upload",
  upload.single("media"),
  async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          error: "No media file uploaded"
        })
      }

      console.log(
        "File received:",
        req.file.originalname
      )

      const form = new FormData()

      form.append(
        "file",
        fs.createReadStream(req.file.path),
        {
          filename: req.file.originalname,
          contentType: req.file.mimetype
        }
      )

      console.log("Sending file to FastAPI...")

      const response = await axios.post(
        "http://127.0.0.1:8000/analyze",
        form,
        {
          headers: form.getHeaders(),
          maxContentLength: Infinity,
          maxBodyLength: Infinity
        }
      )

      console.log("FastAPI response received")

      res.json({
        success: true,
        node: {
          filename: req.file.filename,
          originalName: req.file.originalname,
          size: req.file.size
        },
        fastapi: response.data
      })

    } catch (error) {
      console.error(
        "FastAPI connection error:",
        error.message
      )

      res.status(500).json({
        success: false,
        error: "FastAPI analysis request failed"
      })
    }
  }
)

app.listen(5000, () => {
  console.log(
    "Node.js server running on http://localhost:5000"
  )
})