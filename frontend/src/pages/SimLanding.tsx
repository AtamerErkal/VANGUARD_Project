import React, { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'

const BASE = import.meta.env.VITE_API_URL ?? ''

export default function SimLanding() {
  const nav     = useNavigate()
  const [creating, setCreating] = useState(false)
  const [joinId,   setJoinId]   = useState('')
  const [error,    setError]    = useState<string | null>(null)

  async function createSession() {
    setCreating(true)
    setError(null)
    try {
      const res = await fetch(`${BASE}/sim/sessions`, { method: 'POST' })
      if (!res.ok) throw new Error(`Server returned ${res.status}`)
      const { session_id } = await res.json()
      nav(`/sim/display/${session_id}`)
    } catch (e) {
      setError(String(e))
      setCreating(false)
    }
  }

  function joinSession() {
    const id = joinId.trim().toUpperCase()
    if (id.length !== 6) { setError('Session ID must be exactly 6 characters'); return }
    nav(`/sim/${id}`)
  }

  return (
    <div className="min-h-screen bg-[#060d19] text-white flex flex-col items-center justify-center gap-10 p-8">
      {/* Title */}
      <div className="text-center">
        <p className="text-xs text-slate-500 tracking-[0.3em] mb-2">ANTHROPIC · PYTORCH · FASTAPI</p>
        <h1 className="text-4xl font-bold tracking-[0.2em] text-cyan-400 mb-1">VANGUARD AI</h1>
        <p className="text-slate-400 text-base tracking-widest">LIVE MULTI-USER TACTICAL SIMULATION</p>
      </div>

      {/* Cards */}
      <div className="flex flex-col md:flex-row gap-6 w-full max-w-2xl">
        {/* Host */}
        <div className="flex-1 bg-slate-900/70 border border-cyan-900/50 rounded-2xl p-6 flex flex-col gap-4 hover:border-cyan-700/60 transition-colors">
          <div className="flex items-center gap-2">
            <span className="text-2xl">🎯</span>
            <h2 className="text-lg font-bold text-cyan-300 tracking-wider">HOST SESSION</h2>
          </div>
          <p className="text-slate-400 text-sm leading-relaxed">
            Create a new simulation session. Share the participant link with your team —
            their tracks will appear on your tactical display in real time.
          </p>
          <ul className="text-xs text-slate-500 space-y-1 mt-1">
            <li>→ Open the tactical MOC display</li>
            <li>→ Share join link via Teams / Slack</li>
            <li>→ Watch tracks appear as they're submitted</li>
          </ul>
          <button
            onClick={createSession}
            disabled={creating}
            className="mt-auto bg-cyan-800 hover:bg-cyan-700 disabled:opacity-50 disabled:cursor-not-allowed
                       text-white font-bold py-3 rounded-xl transition-colors tracking-wider text-sm"
          >
            {creating ? '⏳ Creating…' : '+ Create New Session'}
          </button>
        </div>

        {/* Join */}
        <div className="flex-1 bg-slate-900/70 border border-green-900/50 rounded-2xl p-6 flex flex-col gap-4 hover:border-green-700/60 transition-colors">
          <div className="flex items-center gap-2">
            <span className="text-2xl">📡</span>
            <h2 className="text-lg font-bold text-green-400 tracking-wider">JOIN SESSION</h2>
          </div>
          <p className="text-slate-400 text-sm leading-relaxed">
            Enter the 6-character session code you received to submit tracks
            to an existing simulation.
          </p>
          <input
            className="bg-slate-800 border border-slate-600 focus:border-green-600 outline-none
                       rounded-xl px-4 py-3 text-center text-2xl font-mono tracking-[0.4em]
                       uppercase text-white placeholder:text-slate-700 transition-colors"
            placeholder="XXXXXX"
            maxLength={6}
            value={joinId}
            onChange={e => { setJoinId(e.target.value.toUpperCase()); setError(null) }}
            onKeyDown={e => e.key === 'Enter' && joinSession()}
          />
          <button
            onClick={joinSession}
            className="mt-auto bg-green-800 hover:bg-green-700 text-white font-bold py-3 rounded-xl
                       transition-colors tracking-wider text-sm"
          >
            Join Session →
          </button>
        </div>
      </div>

      {error && (
        <p className="text-red-400 text-sm bg-red-950/40 border border-red-800/50 rounded-lg px-4 py-2">
          ⚠ {error}
        </p>
      )}

      <Link to="/" className="text-slate-600 hover:text-slate-400 text-sm transition-colors">
        ← Back to VANGUARD AI
      </Link>
    </div>
  )
}
