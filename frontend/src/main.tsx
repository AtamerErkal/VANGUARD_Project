import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import App         from './App'
import SimLanding  from './pages/SimLanding'
import SimDisplay  from './pages/SimDisplay'
import SimSubmit   from './pages/SimSubmit'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        {/* Main VANGUARD AI app */}
        <Route path="/" element={<App />} />

        {/* Simulation — landing (create / join) */}
        <Route path="/sim" element={<SimLanding />} />

        {/* Simulation — operator MOC display */}
        <Route path="/sim/display/:sessionId" element={<SimDisplay />} />

        {/* Simulation — participant track submission form */}
        <Route path="/sim/:sessionId" element={<SimSubmit />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
)
