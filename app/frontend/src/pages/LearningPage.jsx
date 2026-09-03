import { useState } from "react"
import "./LearningPage.css"

const questions = [
  {
    question: "Which signal can help detect an audio deepfake?",
    options: [
      "Audio characteristics",
      "File name only",
      "Video resolution only",
      "File size only"
    ],
    answer: "Audio characteristics"
  },
  {
    question: "What can visual analysis examine?",
    options: [
      "Face manipulation",
      "File name",
      "Folder location",
      "Internet speed"
    ],
    answer: "Face manipulation"
  },
  {
    question: "Why is cross-modal analysis useful?",
    options: [
      "It compares different media signals",
      "It changes the file format",
      "It reduces file size",
      "It renames the file"
    ],
    answer: "It compares different media signals"
  }
]

function LearningPage() {
  const [currentQuestion, setCurrentQuestion] = useState(0)
  const [score, setScore] = useState(0)
  const [selectedAnswer, setSelectedAnswer] = useState(null)
  const [finished, setFinished] = useState(false)

  const question = questions[currentQuestion]

  const handleAnswer = (answer) => {
    if (selectedAnswer) return

    setSelectedAnswer(answer)

    if (answer === question.answer) {
      setScore((previous) => previous + 1)
    }
  }

  const nextQuestion = () => {
    if (currentQuestion === questions.length - 1) {
      setFinished(true)
      return
    }

    setCurrentQuestion((previous) => previous + 1)
    setSelectedAnswer(null)
  }

  const restartQuiz = () => {
    setCurrentQuestion(0)
    setScore(0)
    setSelectedAnswer(null)
    setFinished(false)
  }

  return (
    <div className="learning-page">
      <div className="learning-container">

        <h1 className="learning-title">
          Adaptive Learning
        </h1>

        <p className="learning-description">
          Learn how to identify manipulated media using AI
          detection evidence.
        </p>

        <div className="lesson-card">
          <h2>Understanding Deepfakes</h2>

          <p>
            Deepfakes can manipulate faces, voices, or both.
            Detection systems can examine visual, audio,
            temporal, and cross-modal signals.
          </p>
        </div>

        <div className="quiz-card">

          {!finished ? (
            <>
              <div className="quiz-header">
                <span className="quiz-progress">
                  Question {currentQuestion + 1} of {questions.length}
                </span>

                <span className="quiz-score">
                  Score: {score}
                </span>
              </div>

              <h2>Quick Quiz</h2>

              <p className="quiz-question">
                {question.question}
              </p>

              <div className="quiz-options">
                {question.options.map((option) => (
                  <button
                    key={option}
                    className={`quiz-option ${
                      selectedAnswer === option ? "selected" : ""
                    }`}
                    onClick={() => handleAnswer(option)}
                  >
                    {option}
                  </button>
                ))}
              </div>

              {selectedAnswer && (
                <>
                  <div className="quiz-feedback">
                    {selectedAnswer === question.answer
                      ? "Correct! 🎉"
                      : `Incorrect. Correct answer: ${question.answer}`}
                  </div>

                  <button
                    className="quiz-next"
                    onClick={nextQuestion}
                  >
                    {currentQuestion === questions.length - 1
                      ? "Finish Quiz"
                      : "Next Question"}
                  </button>
                </>
              )}
            </>
          ) : (
            <div className="quiz-finished">
              <h2>Quiz Complete 🎉</h2>

              <p>
                Your score:{" "}
                <strong>
                  {score} / {questions.length}
                </strong>
              </p>

              <button
                className="quiz-next"
                onClick={restartQuiz}
              >
                Try Again
              </button>
            </div>
          )}

        </div>

      </div>
    </div>
  )
}

export default LearningPage