import { BrowserRouter, Routes, Route } from "react-router-dom"
import Navbar from "./components/Navbar"
import Dashboard from "./pages/Dashboard"
import Analyze from "./pages/Analyze"
import AnalysisResult from "./pages/AnalysisResult"
import HistoryPage from "./pages/HistoryPage"
import LearningPage from "./pages/LearningPage"
import "./App.css"

function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/analyze" element={<Analyze />} />
            <Route path="/result" element={<AnalysisResult />} />
            <Route path="/history" element={<HistoryPage />} />
            <Route path="/learning" element={<LearningPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App
