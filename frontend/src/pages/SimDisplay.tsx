import React, { useEffect, useRef, useState, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import type { SimTrack } from '../sim-types'
import { CLASS_STYLES } from '../types'

const BASE = import.meta.env.VITE_API_URL ?? ''

// ── WebSocket URL ─────────────────────────────────────────────────────────────
function getWsUrl(sid: string): string {
  if (BASE) return BASE.replace(/^https:\/\//, 'wss://').replace(/^http:\/\//, 'ws://') + `/sim/${sid}/ws`
  return `ws://${window.location.host}/sim/${sid}/ws`
}

// ── Coord transform  tactical x,y → SVG pixels ───────────────────────────────
function toSvg(pos: { x: number; y: number }) {
  return { sx: 500 + pos.x * 5, sy: 300 + pos.y * 5 }
}

// ── Azimuth spokes ────────────────────────────────────────────────────────────
const SPOKES = [0,45,90,135,180,225,270,315].map(deg => {
  const r = (deg - 90) * Math.PI / 180
  return { x2: 500 + 330 * Math.cos(r), y2: 300 + 330 * Math.sin(r) }
})

// ── Demo tracks (pre-populated) ───────────────────────────────────────────────
const DEMO_TRACKS: SimTrack[] = [
  {
    track_id: 'SIM-H001', submitted_at: 'DEMO', ai_class: 'HOSTILE', ai_conf: 0.91,
    ai_probs: { HOSTILE:0.91, SUSPECT:0.05, UNKNOWN:0.02, NEUTRAL:0.01, 'ASSUMED FRIEND':0.005, FRIEND:0.005 },
    pos: { x: -62, y: -8 },
    sensor_votes: {
      radar: { label:'Radar', icon:'📡', vote:'HOSTILE', conf:0.88, reading:'RCS 0.02 m² · 920 kts' },
      esm:   { label:'ESM',   icon:'📻', vote:'HOSTILE', conf:0.93, reading:'HOSTILE JAMMING' },
      irst:  { label:'IRST',  icon:'🌡️', vote:'SUSPECT', conf:0.76, reading:'High thermal' },
      iff:   { label:'IFF',   icon:'🔑', vote:'SUSPECT', conf:0.84, reading:'NO RESPONSE' },
    },
    fusion: { best:'HOSTILE', probs:{ HOSTILE:0.91, SUSPECT:0.05, UNKNOWN:0.02, NEUTRAL:0.01, 'ASSUMED FRIEND':0.005, FRIEND:0.005 }, weights:{} },
    xai: [],
    altitude_ft:500, speed_kts:920, rcs_m2:0.02, heading:90,
    esm_signature:'HOSTILE_JAMMING', iff_mode:'NO_RESPONSE', flight_profile:'LOW_ALTITUDE_FLYING', weather:'Clear', thermal_signature:'High',
  },
  {
    track_id: 'SIM-S002', submitted_at: 'DEMO', ai_class: 'SUSPECT', ai_conf: 0.78,
    ai_probs: { HOSTILE:0.12, SUSPECT:0.78, UNKNOWN:0.06, NEUTRAL:0.02, 'ASSUMED FRIEND':0.01, FRIEND:0.01 },
    pos: { x: -52, y: -28 },
    sensor_votes: {
      radar: { label:'Radar', icon:'📡', vote:'SUSPECT', conf:0.72, reading:'RCS 0.8 m² · 580 kts' },
      esm:   { label:'ESM',   icon:'📻', vote:'SUSPECT', conf:0.81, reading:'NOISE JAMMING' },
      irst:  { label:'IRST',  icon:'🌡️', vote:'SUSPECT', conf:0.68, reading:'Medium thermal' },
      iff:   { label:'IFF',   icon:'🔑', vote:'SUSPECT', conf:0.84, reading:'NO RESPONSE' },
    },
    fusion: { best:'SUSPECT', probs:{ HOSTILE:0.12, SUSPECT:0.78, UNKNOWN:0.06, NEUTRAL:0.02, 'ASSUMED FRIEND':0.01, FRIEND:0.01 }, weights:{} },
    xai: [],
    altitude_ft:8000, speed_kts:580, rcs_m2:0.8, heading:115,
    esm_signature:'NOISE_JAMMING', iff_mode:'NO_RESPONSE', flight_profile:'AGGRESSIVE_MANEUVERS', weather:'Cloudy', thermal_signature:'Medium',
  },
  {
    track_id: 'SIM-U003', submitted_at: 'DEMO', ai_class: 'UNKNOWN', ai_conf: 0.62,
    ai_probs: { HOSTILE:0.08, SUSPECT:0.14, UNKNOWN:0.62, NEUTRAL:0.10, 'ASSUMED FRIEND':0.04, FRIEND:0.02 },
    pos: { x: 8, y: -57 },
    sensor_votes: {
      radar: { label:'Radar', icon:'📡', vote:'UNKNOWN', conf:0.55, reading:'RCS 1.5 m² · 460 kts' },
      esm:   { label:'ESM',   icon:'📻', vote:'UNKNOWN', conf:0.52, reading:'UNKNOWN EMISSION' },
      irst:  { label:'IRST',  icon:'🌡️', vote:'NEUTRAL', conf:0.48, reading:'Low thermal' },
      iff:   { label:'IFF',   icon:'🔑', vote:'UNKNOWN', conf:0.55, reading:'DEGRADED' },
    },
    fusion: { best:'UNKNOWN', probs:{ HOSTILE:0.08, SUSPECT:0.14, UNKNOWN:0.62, NEUTRAL:0.10, 'ASSUMED FRIEND':0.04, FRIEND:0.02 }, weights:{} },
    xai: [],
    altitude_ft:28000, speed_kts:460, rcs_m2:1.5, heading:180,
    esm_signature:'UNKNOWN_EMISSION', iff_mode:'DEGRADED', flight_profile:'STABLE_CRUISE', weather:'Cloudy', thermal_signature:'Low',
  },
  {
    track_id: 'SIM-N004', submitted_at: 'DEMO', ai_class: 'NEUTRAL', ai_conf: 0.74,
    ai_probs: { HOSTILE:0.02, SUSPECT:0.05, UNKNOWN:0.12, NEUTRAL:0.74, 'ASSUMED FRIEND':0.05, FRIEND:0.02 },
    pos: { x: -5, y: 42 },
    sensor_votes: {
      radar: { label:'Radar', icon:'📡', vote:'NEUTRAL', conf:0.70, reading:'RCS 15 m² · 440 kts' },
      esm:   { label:'ESM',   icon:'📻', vote:'NEUTRAL', conf:0.72, reading:'CLEAN' },
      irst:  { label:'IRST',  icon:'🌡️', vote:'NEUTRAL', conf:0.65, reading:'Low thermal' },
      iff:   { label:'IFF',   icon:'🔑', vote:'ASSUMED FRIEND', conf:0.88, reading:'IFF MODE 3C' },
    },
    fusion: { best:'NEUTRAL', probs:{ HOSTILE:0.02, SUSPECT:0.05, UNKNOWN:0.12, NEUTRAL:0.74, 'ASSUMED FRIEND':0.05, FRIEND:0.02 }, weights:{} },
    xai: [],
    altitude_ft:36000, speed_kts:440, rcs_m2:15, heading:270,
    esm_signature:'CLEAN', iff_mode:'IFF_MODE_3C', flight_profile:'STABLE_CRUISE', weather:'Clear', thermal_signature:'Low',
  },
  {
    track_id: 'SIM-A005', submitted_at: 'DEMO', ai_class: 'ASSUMED FRIEND', ai_conf: 0.83,
    ai_probs: { HOSTILE:0.01, SUSPECT:0.02, UNKNOWN:0.05, NEUTRAL:0.09, 'ASSUMED FRIEND':0.83, FRIEND:0.00 },
    pos: { x: 56, y: -20 },
    sensor_votes: {
      radar: { label:'Radar', icon:'📡', vote:'ASSUMED FRIEND', conf:0.79, reading:'RCS 3 m² · 400 kts' },
      esm:   { label:'ESM',   icon:'📻', vote:'NEUTRAL', conf:0.72, reading:'CLEAN' },
      irst:  { label:'IRST',  icon:'🌡️', vote:'NEUTRAL', conf:0.65, reading:'Medium thermal' },
      iff:   { label:'IFF',   icon:'🔑', vote:'ASSUMED FRIEND', conf:0.88, reading:'IFF MODE 3C' },
    },
    fusion: { best:'ASSUMED FRIEND', probs:{ HOSTILE:0.01, SUSPECT:0.02, UNKNOWN:0.05, NEUTRAL:0.09, 'ASSUMED FRIEND':0.83, FRIEND:0.00 }, weights:{} },
    xai: [],
    altitude_ft:22000, speed_kts:400, rcs_m2:3, heading:270,
    esm_signature:'CLEAN', iff_mode:'IFF_MODE_3C', flight_profile:'STABLE_CRUISE', weather:'Clear', thermal_signature:'Medium',
  },
  {
    track_id: 'SIM-F006', submitted_at: 'DEMO', ai_class: 'FRIEND', ai_conf: 0.96,
    ai_probs: { HOSTILE:0.005, SUSPECT:0.005, UNKNOWN:0.01, NEUTRAL:0.02, 'ASSUMED FRIEND':0.00, FRIEND:0.96 },
    pos: { x: 64, y: 18 },
    sensor_votes: {
      radar: { label:'Radar', icon:'📡', vote:'FRIEND', conf:0.90, reading:'RCS 2 m² · 520 kts' },
      esm:   { label:'ESM',   icon:'📻', vote:'NEUTRAL', conf:0.72, reading:'CLEAN' },
      irst:  { label:'IRST',  icon:'🌡️', vote:'NEUTRAL', conf:0.65, reading:'Medium thermal' },
      iff:   { label:'IFF',   icon:'🔑', vote:'FRIEND', conf:0.98, reading:'IFF MODE 5' },
    },
    fusion: { best:'FRIEND', probs:{ HOSTILE:0.005, SUSPECT:0.005, UNKNOWN:0.01, NEUTRAL:0.02, 'ASSUMED FRIEND':0.00, FRIEND:0.96 }, weights:{} },
    xai: [],
    altitude_ft:18000, speed_kts:520, rcs_m2:2, heading:90,
    esm_signature:'CLEAN', iff_mode:'IFF_MODE_5', flight_profile:'CLIMBING', weather:'Clear', thermal_signature:'Medium',
  },
]

// ── Commercial route definitions ──────────────────────────────────────────────
const COMMERCIAL_ROUTES = [
  { id:'N1', x1:900, y1: 95, x2:100, y2:115 },
  { id:'C1', x1:100, y1:295, x2:900, y2:280 },
  { id:'S1', x1:900, y1:490, x2:100, y2:508 },
]
const COMMERCIAL_AC = [
  { id:'TK401',  route:'N1', t:0.18, label:'TK-401 FL370', flip: false },
  { id:'BA156',  route:'N1', t:0.52, label:'BA-156 FL360', flip: false },
  { id:'LH892',  route:'N1', t:0.83, label:'LH-892 FL350', flip: false },
  { id:'EK215',  route:'C1', t:0.22, label:'EK-215 FL390', flip: true  },
  { id:'TK789',  route:'C1', t:0.65, label:'TK-789 FL380', flip: true  },
  { id:'AA102',  route:'S1', t:0.15, label:'AA-102 FL340', flip: false },
  { id:'UA567',  route:'S1', t:0.58, label:'UA-567 FL350', flip: false },
]

function acPos(routeId: string, t: number) {
  const r = COMMERCIAL_ROUTES.find(r => r.id === routeId)!
  return { x: r.x1 + (r.x2 - r.x1) * t, y: r.y1 + (r.y2 - r.y1) * t }
}

// ── Track symbol ──────────────────────────────────────────────────────────────
function TrackSymbol({ cls }: { cls: string }) {
  const color = CLASS_STYLES[cls]?.color ?? '#888'
  const S = 8
  switch (cls) {
    case 'HOSTILE':        return <polygon points={`0,${S} ${S},${-S} ${-S},${-S}`} fill={color} />
    case 'SUSPECT':        return <polygon points={`0,${-S} ${S},0 0,${S} ${-S},0`} fill="none" stroke={color} strokeWidth={2.5} />
    case 'UNKNOWN':        return <circle r={S} fill="none" stroke={color} strokeWidth={2.5} />
    case 'NEUTRAL':        return <rect x={-S} y={-S} width={S*2} height={S*2} fill="none" stroke={color} strokeWidth={2} />
    case 'ASSUMED FRIEND': return <rect x={-S} y={-S} width={S*2} height={S*2} fill={color} opacity={0.65} />
    case 'FRIEND':         return <polygon points={`0,${-S} ${S},${S} ${-S},${S}`} fill={color} />
    default:               return <circle r={S} fill={color} opacity={0.5} />
  }
}

// ── Inspect modal ─────────────────────────────────────────────────────────────
function InspectModal({ track, onClose }: { track: SimTrack; onClose: () => void }) {
  const s = CLASS_STYLES[track.ai_class]

  // Click outside to close
  const backdropRef = useRef<HTMLDivElement>(null)
  const handleBackdrop = (e: React.MouseEvent) => { if (e.target === backdropRef.current) onClose() }

  const fusionProbs = track.ai_probs ?? track.fusion?.probs ?? {}
  const sortedProbs = Object.entries(fusionProbs).sort(([,a],[,b]) => b - a)

  const xaiItems = (track.xai ?? [])
    .filter(x => x.importance > 0.02)
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 8)

  return (
    <div
      ref={backdropRef}
      onClick={handleBackdrop}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
    >
      <div
        className="w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-2xl border-2 shadow-2xl"
        style={{ borderColor: s?.color + '80', backgroundColor: '#0a1628' }}
      >
        {/* Modal header */}
        <div className="sticky top-0 flex items-center gap-4 px-5 py-4 border-b border-white/10 z-10"
          style={{ backgroundColor: s?.bg ?? '#111' }}>
          <span className="text-4xl">{s?.icon}</span>
          <div className="flex-1">
            <p className="text-[10px] tracking-[0.2em] text-slate-400">AI CLASSIFICATION — {track.track_id}</p>
            <p className="text-2xl font-bold font-mono tracking-wider" style={{ color: s?.color }}>
              {track.ai_class}
            </p>
          </div>
          <div className="text-right mr-4">
            <p className="text-[10px] text-slate-400 tracking-wider">CONFIDENCE</p>
            <p className="text-3xl font-bold font-mono" style={{ color: s?.color }}>
              {Math.round(track.ai_conf * 100)}%
            </p>
          </div>
          <button onClick={onClose}
            className="text-slate-500 hover:text-white text-xl w-8 h-8 flex items-center justify-center
                       rounded-lg hover:bg-white/10 transition-colors ml-2">✕</button>
        </div>

        <div className="p-5 space-y-5">

          {/* Kinematic */}
          <section>
            <h4 className="text-[10px] font-bold text-cyan-500 tracking-[0.2em] mb-2">📊 KINEMATIC DATA</h4>
            <div className="grid grid-cols-3 gap-2 text-xs">
              {[
                { label:'Altitude', val:`${track.altitude_ft.toLocaleString()} ft` },
                { label:'Speed',    val:`${Math.round(track.speed_kts)} kts` },
                { label:'Heading',  val:`${track.heading}°` },
                { label:'RCS',      val:`${track.rcs_m2} m²` },
                { label:'Profile',  val:track.flight_profile.replace(/_/g,' ') },
                { label:'Weather',  val:track.weather },
              ].map(({ label, val }) => (
                <div key={label} className="bg-slate-900/60 border border-slate-800 rounded-lg p-2.5">
                  <p className="text-[9px] text-slate-500 mb-0.5">{label}</p>
                  <p className="font-mono text-cyan-300 font-semibold">{val}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Electronic */}
          <section>
            <h4 className="text-[10px] font-bold text-orange-400 tracking-[0.2em] mb-2">📻 ELECTRONIC SIGNATURE</h4>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-2.5">
                <p className="text-[9px] text-slate-500 mb-0.5">ESM</p>
                <p className="font-mono text-orange-300 font-semibold">{track.esm_signature.replace(/_/g,' ')}</p>
              </div>
              <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-2.5">
                <p className="text-[9px] text-slate-500 mb-0.5">IFF Mode</p>
                <p className="font-mono text-blue-300 font-semibold">{track.iff_mode.replace(/_/g,' ')}</p>
              </div>
            </div>
          </section>

          {/* Sensor Fusion */}
          <section>
            <h4 className="text-[10px] font-bold text-purple-400 tracking-[0.2em] mb-3">🔀 SENSOR FUSION</h4>
            <div className="space-y-2">
              {Object.values(track.sensor_votes ?? {}).map(sv => {
                const sc = CLASS_STYLES[sv.vote]
                return (
                  <div key={sv.label} className="flex items-center gap-3 text-xs bg-slate-900/40 rounded-lg px-3 py-2">
                    <span className="text-base w-6">{sv.icon}</span>
                    <span className="text-slate-400 w-10 font-bold">{sv.label}</span>
                    <span className="text-slate-500 flex-1 text-[10px] font-mono">{sv.reading}</span>
                    <div className="w-24 h-1.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width:`${sv.conf*100}%`, backgroundColor:sc?.color ?? '#888' }} />
                    </div>
                    <span className="text-[10px] font-mono font-bold w-12 text-right" style={{ color:sc?.color }}>
                      {sv.vote} {Math.round(sv.conf*100)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </section>

          {/* Class probabilities */}
          <section>
            <h4 className="text-[10px] font-bold text-slate-400 tracking-[0.2em] mb-3">📈 CLASS PROBABILITIES</h4>
            <div className="space-y-1.5">
              {sortedProbs.map(([cls, prob]) => {
                const cs = CLASS_STYLES[cls]
                return (
                  <div key={cls} className="flex items-center gap-2 text-xs">
                    <span className="w-6 text-sm text-center">{cs?.icon}</span>
                    <span className="w-28 font-mono text-[10px] text-right" style={{ color:cs?.color }}>{cls}</span>
                    <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full transition-all"
                        style={{ width:`${prob*100}%`, backgroundColor:cs?.color }} />
                    </div>
                    <span className="w-9 font-mono font-bold text-[11px] text-right" style={{ color:cs?.color }}>
                      {(prob*100).toFixed(0)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </section>

          {/* XAI */}
          {xaiItems.length > 0 && (
            <section>
              <h4 className="text-[10px] font-bold text-green-400 tracking-[0.2em] mb-3">🧠 EXPLAINABLE AI — WHY THIS CLASSIFICATION?</h4>
              <div className="space-y-2">
                {xaiItems.map(item => {
                  const dir = item.direction
                  const barColor = dir === 'supporting' ? '#22c55e' : dir === 'conflicting' ? '#ef4444' : '#94a3b8'
                  const pct = Math.min(100, item.importance * 100 * 8)
                  return (
                    <div key={item.feature} className="flex items-center gap-2 text-xs bg-slate-900/40 rounded-lg px-3 py-2">
                      <span className="text-[10px]">{dir === 'supporting' ? '✅' : dir === 'conflicting' ? '❌' : '➖'}</span>
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-slate-300 font-semibold text-[11px]">{item.label}</span>
                          <span className="text-slate-500 font-mono text-[10px]">{item.value}</span>
                        </div>
                        <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{ width:`${pct}%`, backgroundColor:barColor }} />
                        </div>
                      </div>
                      <span className="w-14 font-mono text-[10px] text-right" style={{ color: barColor }}>
                        {dir === 'supporting' ? '+' : dir === 'conflicting' ? '-' : ''}
                        {(item.delta * 100).toFixed(1)}%
                      </span>
                    </div>
                  )
                })}
              </div>
              <p className="text-[9px] text-slate-600 mt-2">
                Importance = confidence shift when feature is neutralised. ✅ supports · ❌ conflicts with classification.
              </p>
            </section>
          )}
          {xaiItems.length === 0 && (
            <section>
              <h4 className="text-[10px] font-bold text-green-400 tracking-[0.2em] mb-2">🧠 EXPLAINABLE AI</h4>
              <p className="text-xs text-slate-600 italic">XAI data not available for demo tracks. Submit a track via the participant form to see feature importance analysis.</p>
            </section>
          )}

          {/* Footer */}
          <div className="pt-2 border-t border-white/10 flex items-center justify-between text-[10px] text-slate-600">
            <span>Submitted: {track.submitted_at}</span>
            <span>Map pos: ({track.pos.x}, {track.pos.y})</span>
            <button onClick={onClose}
              className="text-slate-500 hover:text-white border border-slate-700 hover:border-slate-500 px-3 py-1.5 rounded-lg transition-colors">
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Alert notification ────────────────────────────────────────────────────────
interface AlertEntry { track: SimTrack; id: number }

function AlertBanner({ alert, onInspect, onDismiss }: {
  alert: AlertEntry
  onInspect: (t: SimTrack) => void
  onDismiss: (id: number) => void
}) {
  const s = CLASS_STYLES[alert.track.ai_class]
  useEffect(() => {
    const t = setTimeout(() => onDismiss(alert.id), 7000)
    return () => clearTimeout(t)
  }, [alert.id, onDismiss])

  return (
    <div
      className="flex items-center gap-3 rounded-xl border px-4 py-3 text-sm shadow-2xl animate-fade-in"
      style={{ borderColor: s?.color + '70', backgroundColor: s?.bg ?? '#111' }}
    >
      <span className="text-2xl animate-bounce">{s?.icon}</span>
      <div className="flex-1">
        <p className="text-[10px] text-slate-400 tracking-widest">NEW CONTACT DETECTED</p>
        <p className="font-bold font-mono" style={{ color: s?.color }}>{alert.track.ai_class}</p>
        <p className="text-[10px] text-slate-500">{alert.track.track_id} · {Math.round(alert.track.ai_conf*100)}% conf</p>
      </div>
      <button
        onClick={() => onInspect(alert.track)}
        className="px-3 py-1.5 rounded-lg text-xs font-bold transition-colors hover:opacity-80"
        style={{ backgroundColor: s?.color + '25', color: s?.color, border: `1px solid ${s?.color}60` }}
      >
        Inspect
      </button>
      <button onClick={() => onDismiss(alert.id)}
        className="text-slate-600 hover:text-white w-5 h-5 flex items-center justify-center text-xs">✕</button>
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────────
type WsState = 'connecting' | 'live' | 'disconnected'

export default function SimDisplay() {
  const { sessionId } = useParams<{ sessionId: string }>()

  const [tracks,       setTracks]       = useState<SimTrack[]>(DEMO_TRACKS)
  const [participants, setParticipants] = useState(0)
  const [wsState,      setWsState]      = useState<WsState>('connecting')
  const [newIds,       setNewIds]       = useState<Set<string>>(new Set())
  const [copied,       setCopied]       = useState(false)
  const [alerts,       setAlerts]       = useState<AlertEntry[]>([])
  const [inspectTrack, setInspect]      = useState<SimTrack | null>(null)
  const alertCounter = useRef(0)

  // ── Load existing session tracks on mount ─────────────────────────────────
  useEffect(() => {
    if (!sessionId) return
    fetch(`${BASE}/sim/${sessionId}/tracks`)
      .then(r => r.json())
      .then(d => {
        if ((d.tracks ?? []).length > 0)
          setTracks([...DEMO_TRACKS, ...(d.tracks ?? [])])
        setParticipants(d.participant_count ?? 0)
      })
      .catch(console.error)
  }, [sessionId])

  // ── WebSocket ─────────────────────────────────────────────────────────────
  useEffect(() => {
    if (!sessionId) return
    const ws = new WebSocket(getWsUrl(sessionId))

    ws.onopen  = () => setWsState('live')
    ws.onclose = () => setWsState('disconnected')
    ws.onerror = () => setWsState('disconnected')
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'new_track') {
          const t: SimTrack = msg.track
          setTracks(prev => [...prev, t])
          setNewIds(prev => new Set([...prev, t.track_id]))
          setTimeout(() => setNewIds(prev => { const n=new Set(prev); n.delete(t.track_id); return n }), 2500)
          // Alert
          const alertId = ++alertCounter.current
          setAlerts(prev => [...prev.slice(-2), { track: t, id: alertId }])
        }
      } catch { /* ignore */ }
    }
    return () => ws.close()
  }, [sessionId])

  const participantLink = `${window.location.origin}/sim/${sessionId}`
  const copyLink = useCallback(() => {
    navigator.clipboard.writeText(participantLink).then(() => { setCopied(true); setTimeout(()=>setCopied(false),2000) })
  }, [participantLink])

  const dismissAlert  = useCallback((id: number) => setAlerts(p => p.filter(a => a.id !== id)), [])
  const realTracks    = tracks.filter(t => !t.track_id.startsWith('SIM-') || !['H001','S002','U003','N004','A005','F006'].some(d=>t.track_id.endsWith(d)))
  const submittedCount = realTracks.length
  const threats       = tracks.filter(t => t.ai_class==='HOSTILE'||t.ai_class==='SUSPECT').length
  const friendly      = tracks.filter(t => t.ai_class==='FRIEND'||t.ai_class==='ASSUMED FRIEND').length

  const wsColor = wsState==='live' ? 'bg-green-400 animate-pulse' : wsState==='connecting' ? 'bg-yellow-400' : 'bg-red-500'

  return (
    <div className="h-screen bg-[#060d19] text-white flex flex-col overflow-hidden select-none">

      {/* ── Top bar ───────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3 px-4 h-12 bg-black/40 border-b border-slate-800/80 shrink-0 text-xs">
        <span className="font-bold tracking-[0.15em] text-cyan-400 hidden sm:block">VANGUARD AI — TACTICAL MOC</span>
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">SESSION</span>
          <span className="font-mono font-bold text-yellow-300 tracking-widest text-sm">{sessionId}</span>
        </div>
        <button onClick={copyLink}
          className={`border rounded px-2 py-0.5 transition-colors ${copied ? 'border-green-500 text-green-400' : 'border-slate-700 text-slate-400 hover:border-slate-500 hover:text-white'}`}>
          {copied ? '✓ Copied!' : '📋 Copy Participant Link'}
        </button>
        <div className="ml-auto flex items-center gap-4 text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${wsColor}`} />
            <span className={wsState==='live'?'text-green-400':wsState==='connecting'?'text-yellow-400':'text-red-400'}>
              {wsState.toUpperCase()}
            </span>
          </span>
          <span>📡 {participants} online</span>
          <span className="text-slate-300 font-semibold">📤 {submittedCount} submitted</span>
          {threats>0  && <span className="text-red-400">⚠ {threats} threats</span>}
          {friendly>0 && <span className="text-green-400">🛡 {friendly} friendly</span>}
          <Link to="/sim" className="text-slate-600 hover:text-slate-400 transition-colors ml-1">← Home</Link>
        </div>
      </div>

      {/* ── Main area ─────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden relative">

        {/* ── SVG Tactical Map ──────────────────────────────────────── */}
        <div className="flex-1 flex items-stretch bg-[#060d19] overflow-hidden relative">
          <svg viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid meet" className="w-full h-full">
            <defs>
              <linearGradient id="zoneEnemy" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%"   stopColor="#6b0000" stopOpacity="0.30" />
                <stop offset="100%" stopColor="#6b0000" stopOpacity="0.04" />
              </linearGradient>
              <linearGradient id="zoneFriend" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%"   stopColor="#002060" stopOpacity="0.04" />
                <stop offset="100%" stopColor="#002060" stopOpacity="0.30" />
              </linearGradient>
              <radialGradient id="shipGlow" cx="50%" cy="50%" r="50%">
                <stop offset="0%"   stopColor="#00ffff" stopOpacity="0.28" />
                <stop offset="100%" stopColor="#00ffff" stopOpacity="0"   />
              </radialGradient>
              <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
              <filter id="trackGlow" x="-80%" y="-80%" width="260%" height="260%">
                <feGaussianBlur stdDeviation="5" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
              </filter>
            </defs>

            {/* Base */}
            <rect width="1000" height="600" fill="#060d19"/>

            {/* ── Territory polygons ──────────────────────────────── */}
            {/* Enemy landmass — western coast */}
            <polygon
              points="0,0 210,0 195,60 220,110 185,165 215,220 180,275 210,330 175,385 205,440 170,495 200,550 185,600 0,600"
              fill="url(#zoneEnemy)" stroke="rgba(180,30,30,0.25)" strokeWidth="1.5"
            />
            {/* Friendly landmass — eastern coast */}
            <polygon
              points="1000,0 790,0 805,60 780,110 815,165 785,220 820,275 790,330 825,385 795,440 830,495 800,550 815,600 1000,600"
              fill="url(#zoneFriend)" stroke="rgba(30,70,180,0.25)" strokeWidth="1.5"
            />
            {/* Territory labels */}
            <text x="78" y="50" textAnchor="middle" fill="rgba(200,50,50,0.40)" fontSize="9" fontFamily="monospace" letterSpacing="2" fontWeight="bold">ENEMY TERRITORY</text>
            <text x="920" y="50" textAnchor="middle" fill="rgba(50,100,200,0.40)" fontSize="9" fontFamily="monospace" letterSpacing="2" fontWeight="bold">FRIENDLY TERRITORY</text>
            {/* Sea label */}
            <text x="500" y="580" textAnchor="middle" fill="rgba(0,150,200,0.18)" fontSize="11" fontFamily="monospace" letterSpacing="4">NEUTRAL SEA ZONE</text>

            {/* ── Commercial airways ───────────────────────────────── */}
            {COMMERCIAL_ROUTES.map(r => (
              <line key={r.id} x1={r.x1} y1={r.y1} x2={r.x2} y2={r.y2}
                stroke="rgba(200,200,255,0.10)" strokeWidth="1.5" strokeDasharray="8,6"/>
            ))}
            {/* Route labels */}
            <text x="880" y="84"  fill="rgba(200,200,255,0.22)" fontSize="8" fontFamily="monospace">N-AIR-1</text>
            <text x="115" y="270" fill="rgba(200,200,255,0.22)" fontSize="8" fontFamily="monospace">C-AIR-1</text>
            <text x="870" y="478" fill="rgba(200,200,255,0.22)" fontSize="8" fontFamily="monospace">S-AIR-1</text>

            {/* Commercial aircraft */}
            {COMMERCIAL_AC.map(ac => {
              const { x, y } = acPos(ac.route, ac.t)
              const angle = ac.flip ? 0 : 180
              return (
                <g key={ac.id} transform={`translate(${x},${y})`} opacity="0.55">
                  <text fontSize="11" textAnchor="middle" dominantBaseline="middle"
                    fill="rgba(200,220,255,0.70)"
                    transform={`rotate(${angle})`}
                    style={{ userSelect:'none' }}
                  >✈</text>
                  <text y="14" fontSize="7" textAnchor="middle" fill="rgba(180,200,240,0.50)" fontFamily="monospace">
                    {ac.label}
                  </text>
                </g>
              )
            })}

            {/* ── Grid ────────────────────────────────────────────── */}
            {[100,200,300,400,600,700,800,900].map(x => (
              <line key={x} x1={x} y1="0" x2={x} y2="600" stroke="rgba(0,160,230,0.04)" strokeWidth="1"/>
            ))}
            {[100,200,400,500].map(y => (
              <line key={y} x1="0" y1={y} x2="1000" y2={y} stroke="rgba(0,160,230,0.04)" strokeWidth="1"/>
            ))}

            {/* Zone separators */}
            <line x1="460" y1="0" x2="460" y2="600" stroke="rgba(200,60,60,0.15)" strokeWidth="1" strokeDasharray="6,5"/>
            <line x1="540" y1="0" x2="540" y2="600" stroke="rgba(60,100,220,0.15)" strokeWidth="1" strokeDasharray="6,5"/>

            {/* Azimuth spokes */}
            {SPOKES.map(({ x2, y2 }, i) => (
              <line key={i} x1="500" y1="300" x2={x2} y2={y2} stroke="rgba(0,160,230,0.05)" strokeWidth="1"/>
            ))}

            {/* Range rings */}
            {[100,200,300].map(r => (
              <circle key={r} cx="500" cy="300" r={r} fill="none"
                stroke="rgba(0,160,230,0.12)" strokeWidth="1" strokeDasharray="4,10"/>
            ))}
            {[{r:100,l:'20NM'},{r:200,l:'40NM'},{r:300,l:'60NM'}].map(({r,l}) => (
              <text key={r} x={500+r+5} y={297} fill="rgba(0,160,230,0.35)" fontSize="9" fontFamily="monospace">{l}</text>
            ))}

            {/* Compass */}
            <text x="500" y="18"  textAnchor="middle" fill="rgba(0,200,255,0.4)" fontSize="9" fontFamily="monospace">N</text>
            <text x="500" y="595" textAnchor="middle" fill="rgba(0,200,255,0.4)" fontSize="9" fontFamily="monospace">S</text>
            <text x="988" y="303" textAnchor="middle" fill="rgba(0,200,255,0.4)" fontSize="9" fontFamily="monospace">E</text>
            <text x="14"  y="303" textAnchor="middle" fill="rgba(0,200,255,0.4)" fontSize="9" fontFamily="monospace">W</text>

            {/* Axis labels */}
            <text x="170" y="26" textAnchor="middle" fill="rgba(210,50,50,0.55)" fontSize="10" fontFamily="monospace" letterSpacing="3" fontWeight="bold">THREAT AXIS</text>
            <text x="830" y="26" textAnchor="middle" fill="rgba(50,100,210,0.55)" fontSize="10" fontFamily="monospace" letterSpacing="3" fontWeight="bold">FRIENDLY AXIS</text>

            {/* Ship */}
            <circle cx="500" cy="300" r="32" fill="url(#shipGlow)"/>
            <g transform="translate(500,300)" filter="url(#glow)">
              <polygon points="0,-11 11,0 0,11 -11,0" fill="rgba(0,240,240,0.7)" stroke="cyan" strokeWidth="1.5"/>
              <text y="22" textAnchor="middle" fill="rgba(0,240,240,0.6)" fontSize="7.5" fontFamily="monospace" letterSpacing="1">OWN SHIP</text>
            </g>

            {/* ── Track symbols ────────────────────────────────────── */}
            {tracks.map(track => {
              const { sx, sy } = toSvg(track.pos)
              const isNew = newIds.has(track.track_id)
              const isDemo = track.submitted_at === 'DEMO'
              const color = CLASS_STYLES[track.ai_class]?.color ?? '#888'
              return (
                <g key={track.track_id}
                   transform={`translate(${sx},${sy})`}
                   filter={isNew ? 'url(#trackGlow)' : undefined}
                   opacity={isDemo ? 0.75 : 1}
                   style={{ cursor: 'pointer' }}
                   onClick={() => setInspect(track)}
                >
                  {isNew && (
                    <circle r="10" fill="none" stroke={color} strokeWidth="1.5" opacity="0">
                      <animate attributeName="r"       values="10;26;10" dur="1.8s" repeatCount="2"/>
                      <animate attributeName="opacity" values="0.9;0;0.9" dur="1.8s" repeatCount="2"/>
                    </circle>
                  )}
                  <TrackSymbol cls={track.ai_class}/>
                  {/* Demo indicator */}
                  {isDemo && <circle r="12" fill="none" stroke={color} strokeWidth="0.5" opacity="0.3" strokeDasharray="2,3"/>}
                  <text y="18" textAnchor="middle" fill={color} fontSize="7.5" fontFamily="monospace" opacity="0.9">{track.track_id}</text>
                  <text y="27" textAnchor="middle" fill={color} fontSize="6.5" fontFamily="monospace" opacity="0.65">{Math.round(track.ai_conf*100)}%</text>
                </g>
              )
            })}
          </svg>

          {/* ── Alert notifications (overlay bottom-left of map) ── */}
          <div className="absolute bottom-4 left-4 flex flex-col gap-2 max-w-xs z-30 pointer-events-none">
            {alerts.map(alert => (
              <div key={alert.id} className="pointer-events-auto">
                <AlertBanner
                  alert={alert}
                  onInspect={(t) => { setInspect(t); dismissAlert(alert.id) }}
                  onDismiss={dismissAlert}
                />
              </div>
            ))}
          </div>
        </div>

        {/* ── Track Feed Sidebar ─────────────────────────────────── */}
        <div className="w-72 bg-slate-950/60 border-l border-slate-800/70 flex flex-col overflow-hidden shrink-0">
          <div className="px-3 py-2.5 border-b border-slate-800/70 flex items-center justify-between">
            <span className="text-[10px] font-bold text-slate-500 tracking-[0.2em]">TRACK FEED</span>
            <div className="flex items-center gap-2 text-[10px] text-slate-600">
              <span>{tracks.length} total</span>
              <span className="text-slate-700">·</span>
              <span>{DEMO_TRACKS.length} demo</span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
            {[...tracks].reverse().map(track => {
              const s = CLASS_STYLES[track.ai_class]
              const isDemo = track.submitted_at === 'DEMO'
              return (
                <button key={track.track_id}
                  onClick={() => setInspect(track)}
                  className="w-full text-left rounded-xl p-2.5 border text-xs transition-all hover:brightness-125 active:scale-[0.98]"
                  style={{ borderColor:(s?.color??'#888')+'35', backgroundColor:(s?.bg??'#111')+'dd' }}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-bold font-mono" style={{ color:s?.color }}>
                      {s?.icon} {track.ai_class}
                    </span>
                    <span className="text-[10px] text-slate-600">
                      {isDemo ? '— DEMO —' : track.submitted_at}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[10px] text-slate-500">{track.track_id}</span>
                    <span className="font-bold font-mono text-[11px]" style={{ color:s?.color }}>
                      {Math.round(track.ai_conf*100)}%
                    </span>
                  </div>
                  <div className="mt-1.5 h-[3px] bg-slate-800 rounded-full overflow-hidden">
                    <div className="h-full rounded-full" style={{ width:`${track.ai_conf*100}%`, backgroundColor:s?.color }}/>
                  </div>
                  <div className="mt-1.5 text-[9px] text-slate-500">
                    {track.altitude_ft.toLocaleString()} ft · {Math.round(track.speed_kts)} kts · ESM: {track.esm_signature.replace(/_/g,' ')}
                  </div>
                  <div className="mt-0.5 text-[9px] text-slate-600">
                    Click to inspect ↗
                  </div>
                </button>
              )
            })}
          </div>

          {/* Demo note */}
          <div className="border-t border-slate-800/70 px-3 py-2">
            <p className="text-[9px] text-slate-700">
              6 demo tracks pre-loaded. Click any track or use the inspect alert on new arrivals.
            </p>
          </div>
        </div>
      </div>

      {/* ── Inspect modal ──────────────────────────────────────────── */}
      {inspectTrack && <InspectModal track={inspectTrack} onClose={() => setInspect(null)}/>}
    </div>
  )
}
