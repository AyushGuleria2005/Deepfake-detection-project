import { Link } from "react-router-dom"
import "./Navbar.css"

function Navbar() {
  return (
    <nav className="navbar">
      <h2 className="navbar-title">DeepFake AI</h2>

      <div className="navbar-links">
        <Link to="/">
          <button className="navbar-button">Dashboard</button>
        </Link>

        <Link to="/analyze">
          <button className="navbar-button">Analyze</button>
        </Link>

        <Link to="/history">
          <button className="navbar-button">History</button>
        </Link>

        <Link to="/learning">
          <button className="navbar-button">Learning</button>
        </Link>
      </div>
    </nav>
  )
}

export default Navbar