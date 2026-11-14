import { useState } from "react"
import "./LearningPage.css"

const questions = [
  {
    category: "Spatial Artifacts",
    question: "What visual artifact is commonly found around the boundary of a face-swap deepfake?",
    options: [
      "Blending seams and inconsistent skin color gradients",
      "Low network audio bitrate",
      "Higher video file frame rate",
      "Loss of video audio channels"
    ],
    answer: "Blending seams and inconsistent skin color gradients",
    explanation: "Deepfake models often generate only facial regions; blending algorithms struggle with hair borders and varying skin tones."
  },
  {
    category: "Temporal Dynamics",
    question: "How does a Temporal BiLSTM model differentiate authentic video from synthetic manipulation?",
    options: [
      "By verifying chronological facial continuity and micro-blink patterns across frames",
      "By analyzing the file creation metadata header",
      "By checking if the video resolution is 4K",
      "By measuring internet download speed"
    ],
    answer: "By verifying chronological facial continuity and micro-blink patterns across frames",
    explanation: "Deepfakes often flicker across frames. Recurrent temporal models detect unnatural variance in landmark distances over time."
  },
  {
    category: "Neural Architectures",
    question: "Why is self-attention valuable in temporal deepfake detection?",
    options: [
      "It focuses the model's weights on anomalous transition frames where artifacts flicker",
      "It compresses the video file size by 50%",
      "It renders 3D avatars directly in real time",
      "It generates replacement audio tracks"
    ],
    answer: "It focuses the model's weights on anomalous transition frames where artifacts flicker",
    explanation: "Attention mechanisms dynamically assign higher weights to key glitch frames, improving classification of subtle manipulations."
  },
  {
    category: "Face Normalization",
    question: "Why is face-crop normalization critical before passing frames into ResNet18?",
    options: [
      "It eliminates background bias and standardizes spatial facial proportions",
      "It reduces GPU memory usage to zero",
      "It turns grayscale frames into color",
      "It automatically enhances video brightness"
    ],
    answer: "It eliminates background bias and standardizes spatial facial proportions",
    explanation: "Cropping to the face prevents the neural network from overfitting to background static or room lighting cues."
  }
]

function LearningPage() {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [score, setScore] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState(null)
  const [finished, setFinished] = useState(false)

  const q = questions[currentQuestion]

  const handleAnswer = (answer) => {
    if (selectedAnswer) return
    setSelectedAnswer(answer)
    if (answer === q.answer) {
      setScore((prev) => prev + 1)
    }
  }

  const nextQuestion = () => {
    if (currentQuestion === questions.length - 1) {
      setFinished(true)
      return
    }
    setCurrentQuestion((prev) => prev + 1)
    setSelectedAnswer(null)
  }

  const restartQuiz = () => {
    setCurrentQuestion(0)
    setScore(0)
    setSelectedAnswer(null)
    setFinished(false)
  }

  const progressPercent = Math.round(((currentQuestion + (selectedAnswer ? 1 : 0)) / questions.length) * 100)

  return (
    <div className="learning-page">
      <div className="learning-header">
        <div className="page-tag">Educational Sandbox</div>
        <h1 className="learning-title">Deepfake Forensics & Verification Academy</h1>
        <p className="learning-description">
          Master the foundational concepts behind spatial CNNs, temporal sequence modeling, and synthetic artifact recognition.
        </p>
      </div>

      {!finished ? (
        <div className="quiz-card">
          <div className="quiz-top">
            <span className="quiz-category font-mono">{q.category}</span>
            <span className="quiz-step font-mono">
              Question {currentQuestion + 1} of {questions.length}
            </span>
          </div>

          <div className="quiz-progress-bar">
            <div
              className="quiz-progress-fill"
              style={{ width: `${progressPercent}%` }}
            ></div>
          </div>

          <h2 className="quiz-question-text">{q.question}</h2>

          <div className="quiz-options-list">
            {q.options.map((option, idx) => {
              let optionClass = ""
              if (selectedAnswer) {
                if (option === q.answer) {
                  optionClass = "correct"
                } else if (option === selectedAnswer) {
                  optionClass = "incorrect"
                }
              }

              return (
                <button
                  key={idx}
                  className={`quiz-option-btn ${optionClass} ${selectedAnswer === option ? "selected" : ""}`}
                  onClick={() => handleAnswer(option)}
                  disabled={selectedAnswer !== null}
                >
                  <span className="option-marker font-mono">
                    {String.fromCharCode(65 + idx)}
                  </span>
                  <span className="option-text">{option}</span>
                  {selectedAnswer && option === q.answer && (
                    <span className="result-indicator">✓</span>
                  )}
                  {selectedAnswer && option === selectedAnswer && option !== q.answer && (
                    <span className="result-indicator">✕</span>
                  )}
                </button>
              )
            })}
          </div>

          {selectedAnswer && (
            <div className="explanation-box">
              <span className="explanation-title">Forensic Note:</span>
              <p>{q.explanation}</p>
            </div>
          )}

          <div className="quiz-footer">
            <span className="current-score font-mono">
              Current Score: {score} / {questions.length}
            </span>
            <button
              className="btn-next"
              onClick={nextQuestion}
              disabled={!selectedAnswer}
            >
              {currentQuestion === questions.length - 1 ? "Complete Quiz" : "Next Question →"}
            </button>
          </div>
        </div>
      ) : (
        <div className="quiz-finished-card">
          <div className="finish-badge">🎓</div>
          <h2>Academy Module Completed</h2>
          <p className="finish-sub">
            You scored <span className="font-mono finish-score">{score}</span> out of <span className="font-mono finish-score">{questions.length}</span> (
            {Math.round((score / questions.length) * 100)}%)
          </p>

          <p className="finish-desc">
            {score === questions.length
              ? "Exemplary performance! You possess comprehensive knowledge of spatial artifacting and temporal deepfake forensics."
              : "Solid attempt. Review spatial blending edges and LSTM coherence to sharpen your forensic detection skills."}
          </p>

          <div className="finish-actions">
            <button className="btn-secondary" onClick={restartQuiz}>
              Retake Forensic Quiz
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default LearningPage
