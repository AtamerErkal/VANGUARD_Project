import React, { useEffect, useRef, useState, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import type { SimTrack } from '../sim-types'
import { CLASS_STYLES } from '../types'

const BASE = import.meta.env.VITE_API_URL ?? ''

// ── WebSocket URL ─────────────────────────────────────────────────────────────
function getWsUrl(sessionId: string): string {
  if (BASE) {
    const wsBase = BASE.replace(/^https:\/\//, 'wss://').replace(/^http:\/\//, 'ws://')
    return `${wsBase}/sim/${sessionId}/ws`
  }
  return `ws://${window.location.host}/sim/${sessionId}/ws`
}

// ── Coordinate transform ──────────────────────────────────────────────────────
// tactical: x∈[-80,80] y∈[-45,45]  →  svg: ship at (500,300), scale ×5
function toSvg(pos: { x: number; y: number }) {
  return { sx: 500 + pos.x * 5, sy: 300 + pos.y * 5 }
}

// ── Track symbol ──────────────────────────────────────────────────────────────
function TrackSymbol({ cls }: { cls: string }) {
  const color = CLASS_STYLES[cls]?.color ?? '#888'
  const S = 8
  switch (cls) {
    case 'HOSTILE':
      // filled inverted triangle — threat from the enemy side
      return <polygon points={`0,${S} ${S},${-S} ${-S},${-S}`} fill={color} />
    case 'SUSPECT':
      // open diamond
      return <polygon points={`0,${-S} ${S},0 0,${S} ${-S},0`}
                fill="none" stroke={color} strokeWidth={2.5} />
    case 'UNKNOWN':
      return <circle r={S} fill="none" stroke={color} strokeWidth={2.5} />
    case 'NEUTRAL':
      return <rect x={-S} y={-S} width={S*2} height={S*2}
               fill="none" stroke={color} strokeWidth={2} />
    case 'ASSUMED FRIEND':
      // half-filled square
      return <rect x={-S} y={-S} width={S*2} height={S*2}
               fill={color} opacity={0.65} />
    case 'FRIEND':
      // filled upright triangle
      return <polygon points={`0,${-S} ${S},${S} ${-S},${S}`} fill={color} />
    default:
      return <circle r={S} fill={color} opacity={0.5} />
  }
}

// ── Azimuth lines (8 spoke) ───────────────────────────────────────────────────
const SPOKES = [0, 45, 90, 135, 180, 225, 270, 315].map(deg => {
  const rad = (deg - 90) * Math.PI / 180
  return { x2: 500 + 330 * Math.cos(rad), y2: 300 + 330 * Math.sin(rad) }
})

type WsState = 'connecting' | 'live' | 'disconnected'

export default function SimDisplay() {
  const { sessionId } = useParams<{ sessionId: string }>()

  const [tracks,      setTracks]      = useState<SimTrack[]>([])
  const [participants,setParticipants]= useState(0)
  const [wsState,     setWsState]     = useState<WsState>('connecting')
  const [newIds,      setNewIds]      = useState<Set<string>>(new Set())
  const [copied,      setCopied]      = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  // ── Load existing tracks on mount ─────────────────────────────────────────
  useEffect(() => {
    if (!sessionId) return
    fetch(`${BASE}/sim/${sessionId}/tracks`)
      .then(r => r.json())
      .then(d => {
        setTracks(d.tracks ?? [])
        setParticipants(d.participant_count ?? 0)
      })
      .catch(console.error)
  }, [sessionId])

  // ── WebSocket ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!sessionId) return
    const ws = new WebSocket(getWsUrl(sessionId))
    wsRef.current = ws

    ws.onopen  = () => setWsState('live')
    ws.onclose = () => setWsState('disconnected')
    ws.onerror = () => setWsState('disconnected')

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'new_track') {
          const track: SimTrack = msg.track
          setTracks(prev => [...prev, track])
          setNewIds(prev => new Set([...prev, track.track_id]))
          setTimeout(() => {
            setNewIds(prev => { const n = new Set(prev); n.delete(track.track_id); return n })
          }, 2500)
        }
      } catch { /* ignore malformed */ }
    }

    return () => ws.close()
  }, [sessionId])

  // ── Copy participant link ─────────────────────────────────────────────────
  const participantLink = `${window.location.origin}/sim/${sessionId}`
  const copyLink = useCallback(() => {
    navigator.clipboard.writeText(participantLink).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }, [participantLink])

  // ── Threat summary ────────────────────────────────────────────────────────
  const threats   = tracks.filter(t => t.ai_class === 'HOSTILE' || t.ai_class === 'SUSPECT').length
  const friendly  = tracks.filter(t => t.ai_class === 'FRIEND'  || t.ai_class === 'ASSUMED FRIEND').length
  const unknown   = tracks.filter(t => t.ai_class === 'UNKNOWN' || t.ai_class === 'NEUTRAL').length

  const wsColor = wsState === 'live' ? 'bg-green-400 animate-pulse' : wsState === 'connecting' ? 'bg-yellow-400' : 'bg-red-500'

  return (
    <div className="h-screen bg-[#060d19] text-white flex flex-col overflow-hidden select-none">

      {/* ── Top Bar ─────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 px-4 h-12 bg-black/40 border-b border-slate-800/80 shrink-0 text-xs">
        <span className="text-cyan-400 font-bold tracking-[0.15em] hidden sm:block">🎯 VANGUARD AI — TACTICAL MOC</span>
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">SESSION</span>
          <span className="font-mono text-yellow-300 font-bold tracking-widest text-sm">{sessionId}</span>
        </div>
        <button
          onClick={copyLink}
          className={`border rounded px-2 py-0.5 transition-colors ${copied ? 'border-green-500 text-green-400' : 'border-slate-700 text-slate-400 hover:border-slate-500 hover:text-white'}`}
        >
          {copied ? '✓ Copied!' : '📋 Copy Participant Link'}
        </button>

        <div className="ml-auto flex items-center gap-4 text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${wsColor}`} />
            <span className={wsState === 'live' ? 'text-green-400' : wsState === 'connecting' ? 'text-yellow-400' : 'text-red-400'}>
              {wsState.toUpperCase()}
            </span>
          </span>
          <span>📡 {participants} online</span>
          <span className="text-slate-300 font-semibold">Σ {tracks.length} tracks</span>
          {threats  > 0 && <span className="text-red-400">⚠ {threats} threats</span>}
          {friendly > 0 && <span className="text-green-400">🛡 {friendly} friendly</span>}
          {unknown  > 0 && <span className="text-purple-400">❓ {unknown} unknown</span>}
          <Link to="/sim" className="text-slate-600 hover:text-slate-400 transition-colors ml-1">← Home</Link>
        </div>
      </div>

      {/* ── Main ────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* ── SVG Tactical Map ─────────────────────────────────────────── */}
        <div className="flex-1 flex items-stretch bg-[#060d19] overflow-hidden">
          <svg
            viewBox="0 0 1000 600"
            preserveAspectRatio="xMidYMid meet"
            className="w-full h-full"
          >
            <defs>
              {/* Zone gradients */}
              <linearGradient id="zoneEnemy" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%"   stopColor="#6b0000" stopOpacity="0.22" />
                <stop offset="100%" stopColor="#6b0000" stopOpacity="0.04" />
              </linearGradient>
              <linearGradient id="zoneFriend" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%"   stopColor="#002060" stopOpacity="0.04" />
                <stop offset="100%" stopColor="#002060" stopOpacity="0.22" />
              </linearGradient>
              {/* Ship glow */}
              <radialGradient id="shipGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%"   stopColor="#00ffff" stopOpacity="0.25" />
                <stop offset="100%" stopColor="#00ffff" stopOpacity="0"   />
              </radialGradient>
              {/* Filters */}
              <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
              <filter id="trackGlow" x="-80%" y="-80%" width="260%" height="260%">
                <feGaussianBlur stdDeviation="5" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>

            {/* Base fill */}
            <rect width="1000" height="600" fill="#060d19" />

            {/* Zone tints */}
            <rect x="0"   y="0" width="460"  height="600" fill="url(#zoneEnemy)"  />
            <rect x="540" y="0" width="460"  height="600" fill="url(#zoneFriend)" />

            {/* Subtle grid */}
            {[100,200,300,400,600,700,800,900].map(x => (
              <line key={x} x1={x} y1="0" x2={x} y2="600"
                stroke="rgba(0,160,230,0.035)" strokeWidth="1" />
            ))}
            {[100,200,400,500].map(y => (
              <line key={y} x1="0" y1={y} x2="1000" y2={y}
                stroke="rgba(0,160,230,0.035)" strokeWidth="1" />
            ))}

            {/* Zone separator dashes */}
            <line x1="460" y1="0" x2="460" y2="600"
              stroke="rgba(200,60,60,0.18)" strokeWidth="1" strokeDasharray="6,5" />
            <line x1="540" y1="0" x2="540" y2="600"
              stroke="rgba(60,100,220,0.18)" strokeWidth="1" strokeDasharray="6,5" />

            {/* Azimuth spokes */}
            {SPOKES.map(({ x2, y2 }, i) => (
              <line key={i} x1="500" y1="300" x2={x2} y2={y2}
                stroke="rgba(0,160,230,0.05)" strokeWidth="1" />
            ))}

            {/* Range rings */}
            {[100, 200, 300].map(r => (
              <circle key={r} cx="500" cy="300" r={r}
                fill="none" stroke="rgba(0,160,230,0.13)"
                strokeWidth="1" strokeDasharray="4,10" />
            ))}

            {/* Range ring labels */}
            {[{r:100,l:'20NM'},{r:200,l:'40NM'},{r:300,l:'60NM'}].map(({r,l}) => (
              <text key={r}
                x={500 + r + 5} y={297}
                fill="rgba(0,160,230,0.35)" fontSize="9" fontFamily="monospace"
              >{l}</text>
            ))}

            {/* Compass labels */}
            <text x="500" y="18"  textAnchor="middle" fill="rgba(0,200,255,0.4)" fontSize="9" fontFamily="monospace">N</text>
            <text x="500" y="595" textAnchor="middle" fill="rgba(0,200,255,0.4)" fontSize="9" fontFamily="monospace">S</text>
            <text x="988" y="303" textAnchor="middle" fill="rgba(0,200,255,0.4)" fontSize="9" fontFamily="monospace">E</text>
            <text x="14"  y="303" textAnchor="middle" fill="rgba(0,200,255,0.4)" fontSize="9" fontFamily="monospace">W</text>

            {/* Zone axis labels */}
            <text x="170" y="26" textAnchor="middle"
              fill="rgba(210,50,50,0.55)" fontSize="10" fontFamily="monospace" letterSpacing="3" fontWeight="bold"
            >THREAT AXIS</text>
            <text x="830" y="26" textAnchor="middle"
              fill="rgba(50,100,210,0.55)" fontSize="10" fontFamily="monospace" letterSpacing="3" fontWeight="bold"
            >FRIENDLY AXIS</text>

            {/* Own ship glow */}
            <circle cx="500" cy="300" r="32" fill="url(#shipGlow)" />

            {/* Own ship symbol — cyan diamond */}
            <g transform="translate(500,300)" filter="url(#glow)">
              <polygon points="0,-11 11,0 0,11 -11,0"
                fill="rgba(0,240,240,0.7)" stroke="cyan" strokeWidth="1.5" />
              <text y="22" textAnchor="middle"
                fill="rgba(0,240,240,0.6)" fontSize="7.5" fontFamily="monospace" letterSpacing="1"
              >OWN SHIP</text>
            </g>

            {/* Track symbols */}
            {tracks.map(track => {
              const { sx, sy } = toSvg(track.pos)
              const isNew = newIds.has(track.track_id)
              const color = CLASS_STYLES[track.ai_class]?.color ?? '#888'
              return (
                <g key={track.track_id}
                   transform={`translate(${sx},${sy})`}
                   filter={isNew ? 'url(#trackGlow)' : undefined}
                >
                  {/* Arrival pulse ring */}
                  {isNew && (
                    <circle r="10" fill="none" stroke={color} strokeWidth="1.5" opacity="0">
                      <animate attributeName="r"       values="10;26;10" dur="1.8s" repeatCount="2" />
                      <animate attributeName="opacity" values="0.9;0;0.9" dur="1.8s" repeatCount="2" />
                    </circle>
                  )}

                  <TrackSymbol cls={track.ai_class} />

                  {/* Track ID */}
                  <text y="18" textAnchor="middle"
                    fill={color} fontSize="7.5" fontFamily="monospace" opacity="0.9"
                  >{track.track_id}</text>

                  {/* Confidence */}
                  <text y="27" textAnchor="middle"
                    fill={color} fontSize="6.5" fontFamily="monospace" opacity="0.65"
                  >{Math.round(track.ai_conf * 100)}%</text>
                </g>
              )
            })}

            {/* Empty-state hint */}
            {tracks.length === 0 && (
              <text x="500" y="330" textAnchor="middle"
                fill="rgba(80,110,150,0.45)" fontSize="12" fontFamily="monospace" letterSpacing="1"
              >Awaiting tracks — share the participant link to begin</text>
            )}
          </svg>
        </div>

        {/* ── Track Feed Sidebar ────────────────────────────────────────── */}
        <div className="w-72 bg-slate-950/60 border-l border-slate-800/70 flex flex-col overflow-hidden shrink-0">

          {/* Sidebar header */}
          <div className="px-3 py-2.5 border-b border-slate-800/70 flex items-center justify-between">
            <span className="text-[10px] font-bold text-slate-500 tracking-[0.2em]">TRACK FEED</span>
            <span className="text-[10px] text-slate-600">{tracks.length} total</span>
          </div>

          {/* Scrollable track list */}
          <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
            {[...tracks].reverse().map(track => {
              const s = CLASS_STYLES[track.ai_class]
              return (
                <div key={track.track_id}
                  className="rounded-xl p-2.5 border text-xs"
                  style={{ borderColor: (s?.color ?? '#888') + '35', backgroundColor: (s?.bg ?? '#111') + 'dd' }}
                >
                  {/* Class + time */}
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold font-mono" style={{ color: s?.color }}>
                      {s?.icon} {track.ai_class}
                    </span>
                    <span className="text-[10px] text-slate-600">{track.submitted_at}</span>
                  </div>

                  {/* Track ID + confidence */}
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[10px] text-slate-500">{track.track_id}</span>
                    <span className="font-bold font-mono text-[11px]" style={{ color: s?.color }}>
                      {Math.round(track.ai_conf * 100)}%
                    </span>
                  </div>

                  {/* Confidence bar */}
                  <div className="mt-1.5 h-[3px] bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full rounded-full"
                      style={{ width: `${track.ai_conf * 100}%`, backgroundColor: s?.color }}
                    />
                  </div>

                  {/* Kinematic summary */}
                  <div className="mt-1.5 text-[10px] text-slate-500 space-y-0.5">
                    <div>{track.altitude_ft.toLocaleString()} ft · {Math.round(track.speed_kts)} kts</div>
                    <div>ESM: {track.esm_signature.replace(/_/g,' ')} · IFF: {track.iff_mode.replace(/_/g,' ')}</div>
                  </div>
                </div>
              )
            })}

            {tracks.length === 0 && (
              <div className="text-center py-8 text-slate-700 text-xs">
                No tracks submitted yet
              </div>
            )}
          </div>

          {/* Legend */}
          <div className="border-t border-slate-800/70 p-3">
            <p className="text-[9px] font-bold text-slate-600 tracking-[0.2em] mb-2">NATO SYMBOLOGY</p>
            <div className="grid grid-cols-2 gap-x-3 gap-y-1">
              {Object.entries(CLASS_STYLES).map(([cls, s]) => (
                <div key={cls} className="flex items-center gap-1.5 text-[10px]">
                  <span style={{ color: s.color }}>{s.icon}</span>
                  <span className="font-mono" style={{ color: s.color }}>{s.nato}</span>
                  <span className="text-slate-600">{cls}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
