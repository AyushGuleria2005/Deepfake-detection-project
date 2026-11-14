import { NavLink, Link } from "react-router-dom"
import "./Navbar.css"

function Navbar() {
  return (
    <header className="navbar-wrapper">
      <div className="navbar-inner">
        <Link to="/" className="navbar-brand">
          <div className="brand-logo">
            <svg width="20" height="20" viewBox="0 0 76 65" fill="none">
              <path d="M37.5274 0L75.0548 65H0L37.5274 0Z" fill="#ffffff" />
            </svg>
          </div>
          <span className="brand-name">DeepFake<span className="brand-badge">Engine</span></span>
        </Link>

        <nav className="navbar-nav">
          <NavLink
            to="/"
            end
            className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
          >
            Dashboard
          </NavLink>
          <NavLink
            to="/analyze"
            className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
          >
            Analyze
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
          >
            History
          </NavLink>
          <NavLink
            to="/learning"
            className={({ isActive }) => `nav-item ${isActive ? "active" : ""}`}
          >
            Learning
          </NavLink>
        </nav>

        <div className="navbar-actions">
          <div className="status-pill">
            <span className="status-dot"></span>
            <span className="status-text">ResNet18 + BiLSTM</span>
          </div>
          <a
            href="http://localhost:8000/docs"
            target="_blank"
            rel="noopener noreferrer"
            className="api-link"
          >
            API Docs
            <span className="arrow">↗</span>
          </a>
        </div>
      </div>
    </header>
  )
}

export default Navbar
