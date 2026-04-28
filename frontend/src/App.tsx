import { useEffect, useState } from 'react'
import type { Track, ApprovalState } from './types'
import { CLASS_STYLES } from './types'
import { api } from './api'
import Header         from './components/Header'
import TacticalMap    from './components/TacticalMap'
import SensorFusion   from './components/SensorFusion'
import AnomalyAlerts  from './components/AnomalyAlerts'
import Trail3D        from './components/Trail3D'
import ExpertApproval from './components/ExpertApproval'
import WhatIfPanel    from './components/WhatIfPanel'
import XAIPanel       from './components/XAIPanel'
import TrackInsights  from './components/TrackInsights'

/** NATO APP-6 tactical symbol rendered as inline SVG. */
function NATOSymbol({ cls, size = 44 }: { cls: string; size?: number }) {
  const s = CLASS_STYLES[cls] ?? CLASS_STYLES['NEUTRAL']
  const label = s.nato
  const fill  = cls === 'HOSTILE' ? '#fff' : s.color

  const shapes: Record<string, JSX.Element> = {
    HOSTILE: (
      <svg width={size} height={size} viewBox="0 0 40 40">
        <polygon points="20,1 39,20 20,39 1,20" fill={s.color} />
        <text x="20" y="26" textAnchor="middle" fill={fill} fontSize="13" fontWeight="800" fontFamily="Orbitron,monospace">{label}</text>
      </svg>
    ),
    SUSPECT: (
      <svg width={size} height={size} viewBox="0 0 40 40">
        <polygon points="20,2 38,20 20,38 2,20" fill={s.bg} stroke={s.color} strokeWidth="2.5" />
        <text x="20" y="26" textAnchor="middle" fill={fill} fontSize="13" fontWeight="800" fontFamily="Orbitron,monospace">{label}</text>
      </svg>
    ),
    UNKNOWN: (
      <svg width={size} height={size} viewBox="0 0 40 40">
        <rect x="3" y="3" width="34" height="34" rx="2" fill={s.bg} stroke={s.color} strokeWidth="2.5" />
        <text x="20" y="26" textAnchor="middle" fill={fill} fontSize="13" fontWeight="800" fontFamily="Orbitron,monospace">{label}</text>
      </svg>
    ),
    NEUTRAL: (
      <svg width={size} height={size} viewBox="0 0 40 40">
        <rect x="3" y="10" width="34" height="20" fill={s.bg} stroke={s.color} strokeWidth="2.5" />
        <text x="20" y="24" textAnchor="middle" fill={fill} fontSize="13" fontWeight="800" fontFamily="Orbitron,monospace">{label}</text>
      </svg>
    ),
    'ASSUMED FRIEND': (
      <svg width={size} height={size} viewBox="0 0 40 40">
        <rect x="3" y="3" width="34" height="34" rx="8" fill={s.bg} stroke={s.color} strokeWidth="2.5" />
        <text x="20" y="26" textAnchor="middle" fill={fill} fontSize="13" fontWeight="800" fontFamily="Orbitron,monospace">{label}</text>
      </svg>
    ),
    FRIEND: (
      <svg width={size} height={size} viewBox="0 0 40 40">
        <circle cx="20" cy="20" r="17" fill={s.bg} stroke={s.color} strokeWidth="2.5" />
        <text x="20" y="26" textAnchor="middle" fill={fill} fontSize="13" fontWeight="800" fontFamily="Orbitron,monospace">{label}</text>
      </svg>
    ),
  }

  return shapes[cls] ?? shapes['UNKNOWN']
}

type PanelTab = 'fusion' | 'xai' | 'trail'

export default function App() {
  const [tracks,     setTracks]     = useState<Track[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [approvals,  setApprovals]  = useState<Record<string, ApprovalState>>({})
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState<string | null>(null)
  const [activeTab,  setActiveTab]  = useState<PanelTab>('fusion')
  const [panelOpen,  setPanelOpen]  = useState(true)

  useEffect(() => {
    api.getTracks()
      .then(data => { setTracks(data); setSelectedId(data[0]?.track_id ?? null) })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => { setActiveTab('fusion') }, [selectedId])

  const selected      = tracks.find(t => t.track_id === selectedId) ?? null
  const pendingCount  = tracks.filter(t => !approvals[t.track_id]).length
  const approvedCount = Object.values(approvals).filter(a => a.action === 'approved').length

  const threatScore = Math.min(100, tracks.reduce((s, t) => {
    if (t.ai_class === 'HOSTILE')  return s + 15
    if (t.ai_class === 'SUSPECT')  return s + 8
    if (t.ai_class === 'UNKNOWN')  return s + 3
    return s
  }, 0))

  const handleDecide = (id: string, state: ApprovalState) =>
    setApprovals(prev => ({ ...prev, [id]: state }))

  if (loading) return (
    <div className="h-screen flex items-center justify-center" style={{ background: '#060a10' }}>
      <div className="text-center space-y-3">
        <div className="text-xl animate-pulse" style={{ fontFamily: 'Orbitron, monospace', color: '#38bdf8', letterSpacing: 5 }}>
          VANGUARD AI
        </div>
        <p className="text-xs tracking-widest" style={{ color: '#475569', fontFamily: 'Orbitron, monospace' }}>
          INITIALIZING SENSOR FUSION...
        </p>
      </div>
    </div>
  )

  if (error) return (
    <div className="h-screen flex items-center justify-center" style={{ background: '#060a10' }}>
      <div className="text-center space-y-2">
        <p style={{ color: '#ef4444', fontFamily: 'Orbitron, monospace', letterSpacing: 3 }}>BACKEND OFFLINE</p>
        <p className="text-xs" style={{ color: '#475569' }}>{error}</p>
        <p className="text-xs mt-2" style={{ color: '#334155' }}>
          Run: <code className="text-sky-400">uvicorn backend.main:app --reload</code>
        </p>
      </div>
    </div>
  )

  const TABS: { id: PanelTab; label: string; icon: string }[] = [
    { id: 'fusion', label: 'Sensor Fusion', icon: '📡' },
    { id: 'xai',    label: 'Explainable AI', icon: '🧠' },
    { id: 'trail',  label: '3D Trail',       icon: '🛰️' },
  ]

  return (
    <div className="h-screen flex flex-col overflow-hidden" style={{ background: '#060a10' }}>
      <Header pendingCount={pendingCount} approvedCount={approvedCount} selected={selected} threatScore={threatScore} />

      <div className="flex flex-1 overflow-hidden relative">
        {/* Map */}
        <div className="p-3 transition-all duration-300"
             style={{ width: panelOpen ? '60%' : '100%', minWidth: 0 }}>
          <TacticalMap tracks={tracks} selectedId={selectedId} approvals={approvals} onSelect={setSelectedId} />
        </div>

        {/* Panel toggle button */}
        <button
          onClick={() => setPanelOpen(o => !o)}
          title={panelOpen ? 'Collapse panel' : 'Expand panel'}
          style={{
            position:   'absolute',
            top:        '50%',
            right:      panelOpen ? 'calc(40% - 14px)' : '0px',
            transform:  'translateY(-50%)',
            zIndex:     20,
            width:      28,
            height:     52,
            background: 'rgba(14,22,40,0.95)',
            border:     '1px solid rgba(56,189,248,0.2)',
            borderRadius: panelOpen ? '8px 0 0 8px' : '8px 0 0 8px',
            cursor:     'pointer',
            display:    'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color:      '#38bdf8',
            fontSize:   13,
            transition: 'right 0.3s',
          }}
        >
          {panelOpen ? '›' : '‹'}
        </button>

        {/* Side panel */}
        {panelOpen && selected && (() => {
          const approval  = approvals[selected.track_id]
          const dispCls   = approval?.action === 'override' ? (approval.override_class ?? selected.ai_class) : selected.ai_class
          const dispStyle = CLASS_STYLES[dispCls] ?? CLASS_STYLES['NEUTRAL']

          return (
            <div
              className="flex flex-col overflow-hidden"
              style={{ width: '40%', borderLeft: '1px solid rgba(56,189,248,0.1)', background: 'rgba(6,10,16,0.92)' }}
            >
              {/* Classification badge */}
              <div className="px-4 pt-4 pb-2 flex-shrink-0">
                <div
                  className="rounded-xl px-4 py-3 result-reveal"
                  style={{
                    background: `linear-gradient(135deg,${dispStyle.bg} 0%,rgba(0,0,0,0.5) 100%)`,
                    border:     `1px solid ${dispStyle.color}60`,
                    boxShadow:  `0 0 32px ${dispStyle.color}22, inset 0 1px 0 ${dispStyle.color}18`,
                  }}
                >
                  <div className="flex items-start gap-3">
                    {/* NATO symbol */}
                    <div className="flex-shrink-0 mt-0.5" style={{ filter: `drop-shadow(0 0 8px ${dispStyle.color}80)` }}>
                      <NATOSymbol cls={dispCls} size={48} />
                    </div>

                    {/* Classification text */}
                    <div className="flex-1 min-w-0">
                      <p style={{ color: '#7dd3fc', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 4, fontWeight: 600 }}>
                        AI CLASSIFICATION · MC DROPOUT
                      </p>
                      <p style={{ color: dispStyle.color, fontFamily: 'Orbitron, monospace', fontSize: 19, fontWeight: 900, letterSpacing: 3, lineHeight: 1.15 }}>
                        {dispCls}
                      </p>
                      <div className="flex items-center gap-2 mt-1.5">
                        <div style={{ flex: 1, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.08)' }}>
                          <div style={{ width: `${selected.ai_conf * 100}%`, height: '100%', borderRadius: 2, background: dispStyle.color, transition: 'width 0.6s' }} />
                        </div>
                        <span style={{ color: dispStyle.color, fontWeight: 800, fontSize: 14, minWidth: 42 }}>
                          {(selected.ai_conf * 100).toFixed(1)}%
                        </span>
                      </div>
                      <p style={{ color: '#64748b', fontSize: 11, marginTop: 4 }}>{selected.track_id}</p>
                    </div>

                    {/* Kinematic summary */}
                    <div className="text-right flex-shrink-0 space-y-0.5" style={{ fontSize: 11, lineHeight: 1.9 }}>
                      <div style={{ color: '#cbd5e1' }}>ESM: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{selected.esm_signature}</span></div>
                      <div style={{ color: '#cbd5e1' }}>IFF: <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{selected.iff_mode}</span></div>
                      <div style={{ color: '#94a3b8' }}>{selected.flight_profile}</div>
                      <div style={{ color: '#94a3b8' }}>
                        <span style={{ color: '#64748b' }}>Alt </span>
                        <span style={{ color: '#cbd5e1' }}>{selected.altitude_ft.toLocaleString()} ft</span>
                      </div>
                      <div style={{ color: '#94a3b8' }}>
                        <span style={{ color: '#64748b' }}>Spd </span>
                        <span style={{ color: '#cbd5e1' }}>{selected.speed_kts.toFixed(0)} kts</span>
                        <span style={{ color: '#64748b' }}> · RCS </span>
                        <span style={{ color: '#cbd5e1' }}>{selected.rcs_m2.toFixed(1)} m²</span>
                      </div>
                      <div style={{ color: '#94a3b8' }}>{selected.weather} · {selected.thermal_signature}</div>
                    </div>
                  </div>

                  {approval && (
                    <p style={{ color: approval.action === 'approved' ? '#4ade80' : '#fbbf24', fontSize: 12, marginTop: 10, fontWeight: 600, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 8 }}>
                      {approval.action === 'approved' ? '✓ Expert approved' : `↺ Overridden → ${approval.override_class}`}
                    </p>
                  )}
                </div>
              </div>

              {/* Track Intelligence (Kalman + Uncertainty + DS) */}
              <div className="px-4 pb-2 flex-shrink-0">
                <TrackInsights track={selected} />
              </div>

              {/* Anomalies */}
              {selected.anomalies.length > 0 && (
                <div className="px-4 pb-2 flex-shrink-0">
                  <AnomalyAlerts anomalies={selected.anomalies} />
                </div>
              )}

              {/* Tab bar */}
              <div className="px-4 pb-2 flex-shrink-0">
                <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'rgba(12,18,30,0.8)', border: '1px solid rgba(56,189,248,0.1)' }}>
                  {TABS.map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className="flex-1 py-1.5 rounded-lg text-xs transition-all"
                      style={{
                        background: activeTab === tab.id ? 'rgba(56,189,248,0.18)' : 'transparent',
                        color:      activeTab === tab.id ? '#38bdf8' : '#64748b',
                        fontFamily: 'Space Grotesk, sans-serif',
                        fontWeight: activeTab === tab.id ? 700 : 400,
                        fontSize:   13,
                        border:     activeTab === tab.id ? '1px solid rgba(56,189,248,0.3)' : '1px solid transparent',
                      }}
                    >
                      {tab.icon} {tab.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Tab content */}
              <div className="flex-1 overflow-y-auto px-4 pb-4 space-y-3 min-h-0">
                {activeTab === 'fusion' && <SensorFusion track={selected} />}
                {activeTab === 'xai' && selected.xai && (
                  <XAIPanel xai={selected.xai} aiClass={selected.ai_class} />
                )}
                {activeTab === 'trail' && <Trail3D track={selected} />}

                <ExpertApproval
                  track={selected}
                  approval={approvals[selected.track_id] ?? null}
                  onDecide={handleDecide}
                />
                <WhatIfPanel track={selected} />
              </div>
            </div>
          )
        })()}
      </div>
    </div>
  )
}
