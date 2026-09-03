import { BrowserRouter, Routes, Route } from "react-router-dom"

import Navbar from "./components/Navbar"

import Dashboard from "./pages/Dashboard"
import Analyze from "./pages/Analyze"
import AnalysisResult from "./pages/AnalysisResult"
import HistoryPage from "./pages/HistoryPage"
import LearningPage from "./pages/LearningPage"

function App() {
  return (
    <BrowserRouter>
      <Navbar />

      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/analyze" element={<Analyze />} />
        <Route path="/result" element={<AnalysisResult />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/learning" element={<LearningPage />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App