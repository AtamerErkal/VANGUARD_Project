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

type PanelTab = 'fusion' | 'xai'

function NATOSymbol({ cls, size = 44 }: { cls: string; size?: number }) {
  const s    = CLASS_STYLES[cls] ?? CLASS_STYLES['NEUTRAL']
  const fill = cls === 'HOSTILE' ? '#fff' : s.color
  const t = (label: string, y = '26') => (
    <text x="20" y={y} textAnchor="middle" fill={fill} fontSize="13" fontWeight="800" fontFamily="Orbitron,monospace">{label}</text>
  )
  const shapes: Record<string, JSX.Element> = {
    'HOSTILE':        <svg width={size} height={size} viewBox="0 0 40 40"><polygon points="20,1 39,20 20,39 1,20" fill={s.color} />{t(s.nato)}</svg>,
    'SUSPECT':        <svg width={size} height={size} viewBox="0 0 40 40"><polygon points="20,2 38,20 20,38 2,20" fill={s.bg} stroke={s.color} strokeWidth="2.5" />{t(s.nato)}</svg>,
    'UNKNOWN':        <svg width={size} height={size} viewBox="0 0 40 40"><rect x="3" y="3" width="34" height="34" rx="2" fill={s.bg} stroke={s.color} strokeWidth="2.5" />{t(s.nato)}</svg>,
    'NEUTRAL':        <svg width={size} height={size} viewBox="0 0 40 40"><rect x="3" y="10" width="34" height="20" fill={s.bg} stroke={s.color} strokeWidth="2.5" />{t(s.nato, '24')}</svg>,
    'ASSUMED FRIEND': <svg width={size} height={size} viewBox="0 0 40 40"><rect x="3" y="3" width="34" height="34" rx="8" fill={s.bg} stroke={s.color} strokeWidth="2.5" />{t(s.nato)}</svg>,
    'FRIEND':         <svg width={size} height={size} viewBox="0 0 40 40"><circle cx="20" cy="20" r="17" fill={s.bg} stroke={s.color} strokeWidth="2.5" />{t(s.nato)}</svg>,
  }
  return shapes[cls] ?? shapes['UNKNOWN']
}

function AccordionSection({ title, badge, defaultOpen = false, accentColor = '#38bdf8', children }: {
  title: string; badge?: string | number; defaultOpen?: boolean; accentColor?: string; children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  const isWarn = typeof badge === 'string' && badge.includes('DETECTED')
  return (
    <div className="flex-shrink-0" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
      <button onClick={() => setOpen(o => !o)} className="w-full flex items-center justify-between px-4 py-2 transition-all"
              style={{ background: open ? `${accentColor}0a` : 'transparent', cursor: 'pointer', borderBottom: open ? '1px solid rgba(255,255,255,0.05)' : 'none' }}>
        <span style={{ color: accentColor, fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, fontWeight: 700 }}>{title}</span>
        <div className="flex items-center gap-2">
          {badge !== undefined && !open && (
            <span style={{
              background: isWarn ? 'rgba(251,191,36,0.12)' : 'rgba(74,222,128,0.08)',
              color: isWarn ? '#fbbf24' : '#4ade80', fontSize: 10, fontWeight: 700,
              padding: '1px 8px', borderRadius: 10,
              border: `1px solid ${isWarn ? 'rgba(251,191,36,0.25)' : 'rgba(74,222,128,0.2)'}`,
            }}>{badge}</span>
          )}
          <span style={{ color: '#475569', fontSize: 11 }}>{open ? '▲' : '▼'}</span>
        </div>
      </button>
      {open && <div className="px-4 pb-3 pt-2">{children}</div>}
    </div>
  )
}

export default function App() {
  const [tracks,     setTracks]     = useState<Track[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [approvals,  setApprovals]  = useState<Record<string, ApprovalState>>({})
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState<string | null>(null)
  const [activeTab,  setActiveTab]  = useState<PanelTab>('fusion')
  const [panelOpen,  setPanelOpen]  = useState(true)
  const [trailOpen,  setTrailOpen]  = useState(false)
  const [whatIfOpen, setWhatIfOpen] = useState(false)

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
    if (t.ai_class === 'HOSTILE') return s + 15
    if (t.ai_class === 'SUSPECT') return s + 8
    if (t.ai_class === 'UNKNOWN') return s + 3
    return s
  }, 0))

  const handleDecide = (id: string, state: ApprovalState) =>
    setApprovals(prev => ({ ...prev, [id]: state }))

  if (loading) return (
    <div className="h-screen flex items-center justify-center" style={{ background: '#060a10' }}>
      <div className="text-center space-y-3">
        <div className="text-xl animate-pulse" style={{ fontFamily: 'Orbitron, monospace', color: '#38bdf8', letterSpacing: 5 }}>VANGUARD AI</div>
        <p className="text-xs tracking-widest" style={{ color: '#475569', fontFamily: 'Orbitron, monospace' }}>INITIALIZING SENSOR FUSION...</p>
      </div>
    </div>
  )

  if (error) return (
    <div className="h-screen flex items-center justify-center" style={{ background: '#060a10' }}>
      <div className="text-center space-y-2">
        <p style={{ color: '#ef4444', fontFamily: 'Orbitron, monospace', letterSpacing: 3 }}>BACKEND OFFLINE</p>
        <p className="text-xs" style={{ color: '#475569' }}>{error}</p>
        <p className="text-xs mt-2" style={{ color: '#334155' }}>Run: <code className="text-sky-400">python run_server.py</code></p>
      </div>
    </div>
  )

  const TABS: { id: PanelTab; label: string }[] = [
    { id: 'fusion', label: 'Sensor Fusion' },
    { id: 'xai',    label: 'Explainable AI' },
  ]

  return (
    <div className="h-screen flex flex-col overflow-hidden" style={{ background: '#060a10' }}>
      <Header pendingCount={pendingCount} approvedCount={approvedCount} selected={selected} threatScore={threatScore} />

      <div className="flex flex-1 overflow-hidden relative">

        {/* Map */}
        <div className="p-3 transition-all duration-300 relative"
             style={{ width: panelOpen ? '60%' : '100%', minWidth: 0 }}>
          <TacticalMap tracks={tracks} selectedId={selectedId} approvals={approvals} onSelect={setSelectedId} />

          {/* What-If floating overlay — top-left of map */}
          {selected && (
            <div style={{ position: 'absolute', top: 20, left: 20, zIndex: 10, maxWidth: 390 }}>
              {!whatIfOpen ? (
                <button onClick={() => setWhatIfOpen(true)} style={{
                  background: 'rgba(6,12,24,0.92)', border: '1px solid rgba(56,189,248,0.35)',
                  borderRadius: 8, padding: '6px 14px', color: '#7dd3fc',
                  fontFamily: 'Orbitron, monospace', fontSize: 10, letterSpacing: 2, fontWeight: 700,
                  cursor: 'pointer', backdropFilter: 'blur(12px)',
                  boxShadow: '0 4px 20px rgba(0,0,0,0.6)',
                }}>
                  WHAT-IF ANALYSIS ›
                </button>
              ) : (
                <div style={{
                  background: 'rgba(6,10,20,0.97)', border: '1px solid rgba(56,189,248,0.22)',
                  borderRadius: 12, backdropFilter: 'blur(20px)',
                  boxShadow: '0 8px 40px rgba(0,0,0,0.75)',
                  maxHeight: 'calc(100vh - 120px)', overflowY: 'auto',
                }}>
                  <div className="flex items-center justify-between px-4 py-2.5"
                       style={{ borderBottom: '1px solid rgba(56,189,248,0.1)', position: 'sticky', top: 0, background: 'rgba(6,10,20,0.98)', zIndex: 1 }}>
                    <span style={{ color: '#7dd3fc', fontFamily: 'Orbitron, monospace', fontSize: 10, letterSpacing: 3, fontWeight: 700 }}>WHAT-IF ANALYSIS</span>
                    <button onClick={() => setWhatIfOpen(false)}
                            style={{ background: 'none', border: 'none', color: '#64748b', fontSize: 16, cursor: 'pointer' }}>✕</button>
                  </div>
                  <div className="p-3"><WhatIfPanel track={selected} /></div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Panel toggle */}
        <button onClick={() => setPanelOpen(o => !o)} title={panelOpen ? 'Collapse' : 'Expand'}
                style={{
                  position: 'absolute', top: '50%',
                  right: panelOpen ? 'calc(40% - 14px)' : '0px',
                  transform: 'translateY(-50%)', zIndex: 20,
                  width: 28, height: 52,
                  background: 'rgba(14,22,40,0.95)', border: '1px solid rgba(56,189,248,0.2)',
                  borderRadius: '8px 0 0 8px', cursor: 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  color: '#38bdf8', fontSize: 13, transition: 'right 0.3s',
                }}>
          {panelOpen ? '›' : '‹'}
        </button>

        {/* Side panel */}
        {panelOpen && selected && (() => {
          const approval  = approvals[selected.track_id]
          const dispCls   = approval?.action === 'override' ? (approval.override_class ?? selected.ai_class) : selected.ai_class
          const dispStyle = CLASS_STYLES[dispCls] ?? CLASS_STYLES['NEUTRAL']
          const anomalyCount = selected.anomalies.length

          return (
            <div className="flex flex-col overflow-hidden"
                 style={{ width: '40%', borderLeft: '1px solid rgba(56,189,248,0.1)', background: 'rgba(6,10,16,0.92)' }}>

              {/* Classification badge */}
              <div className="px-4 pt-3 pb-2 flex-shrink-0">
                <div className="rounded-xl px-4 py-3"
                     style={{
                       background: `linear-gradient(135deg,${dispStyle.bg} 0%,rgba(0,0,0,0.5) 100%)`,
                       border: `1px solid ${dispStyle.color}60`,
                       boxShadow: `0 0 28px ${dispStyle.color}20, inset 0 1px 0 ${dispStyle.color}18`,
                     }}>
                  <div className="flex items-start gap-3">
                    <div className="flex-shrink-0 mt-0.5" style={{ filter: `drop-shadow(0 0 8px ${dispStyle.color}80)` }}>
                      <NATOSymbol cls={dispCls} size={46} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p style={{ color: '#7dd3fc', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 4, fontWeight: 600 }}>AI CLASSIFICATION · MC DROPOUT</p>
                      <p style={{ color: dispStyle.color, fontFamily: 'Orbitron, monospace', fontSize: 18, fontWeight: 900, letterSpacing: 3 }}>{dispCls}</p>
                      <div className="flex items-center gap-2 mt-1.5">
                        <div style={{ flex: 1, height: 4, borderRadius: 2, background: 'rgba(255,255,255,0.08)' }}>
                          <div style={{ width: `${selected.ai_conf * 100}%`, height: '100%', borderRadius: 2, background: dispStyle.color }} />
                        </div>
                        <span style={{ color: dispStyle.color, fontWeight: 800, fontSize: 14, minWidth: 42 }}>{(selected.ai_conf * 100).toFixed(1)}%</span>
                      </div>
                      <p style={{ color: '#64748b', fontSize: 11, marginTop: 3 }}>{selected.track_id}</p>
                    </div>
                    <div className="text-right flex-shrink-0" style={{ fontSize: 11, lineHeight: 1.9 }}>
                      <div><span style={{ color: '#64748b' }}>ESM </span><span style={{ color: '#e2e8f0', fontWeight: 600 }}>{selected.esm_signature}</span></div>
                      <div><span style={{ color: '#64748b' }}>IFF </span><span style={{ color: '#e2e8f0', fontWeight: 600 }}>{selected.iff_mode}</span></div>
                      <div style={{ color: '#94a3b8' }}>{selected.flight_profile}</div>
                      <div><span style={{ color: '#64748b' }}>Alt </span><span style={{ color: '#cbd5e1' }}>{selected.altitude_ft.toLocaleString()} ft</span></div>
                      <div><span style={{ color: '#64748b' }}>Spd </span><span style={{ color: '#cbd5e1' }}>{selected.speed_kts.toFixed(0)} kts · RCS {selected.rcs_m2.toFixed(1)} m²</span></div>
                    </div>
                  </div>
                  {approval && (
                    <p style={{ color: approval.action === 'approved' ? '#4ade80' : '#fbbf24', fontSize: 11, marginTop: 8, fontWeight: 600, borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: 7 }}>
                      {approval.action === 'approved' ? '✓ Expert approved' : `↺ Overridden → ${approval.override_class}`}
                    </p>
                  )}
                </div>
              </div>

              {/* Scrollable accordion content */}
              <div className="flex-1 overflow-y-auto min-h-0">

                <AccordionSection title="TRACK INTELLIGENCE" accentColor="#7dd3fc" defaultOpen={false}>
                  <TrackInsights track={selected} />
                </AccordionSection>

                <AccordionSection
                  title="ANOMALY DETECTION"
                  badge={anomalyCount > 0 ? `${anomalyCount} DETECTED` : 'CLEAR'}
                  accentColor={anomalyCount > 0 ? '#fbbf24' : '#4ade80'}
                  defaultOpen={anomalyCount > 0}
                >
                  <AnomalyAlerts anomalies={selected.anomalies} />
                </AccordionSection>

                {/* Tab bar */}
                <div className="px-4 py-2" style={{ borderTop: '1px solid rgba(255,255,255,0.05)' }}>
                  <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'rgba(12,18,30,0.8)', border: '1px solid rgba(56,189,248,0.1)' }}>
                    {TABS.map(tab => (
                      <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                              className="flex-1 py-1.5 rounded-lg transition-all"
                              style={{
                                background: activeTab === tab.id ? 'rgba(56,189,248,0.18)' : 'transparent',
                                color:      activeTab === tab.id ? '#7dd3fc' : '#64748b',
                                fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 1.5,
                                fontWeight: activeTab === tab.id ? 700 : 400,
                                border:     activeTab === tab.id ? '1px solid rgba(56,189,248,0.3)' : '1px solid transparent',
                                cursor:     'pointer',
                              }}>
                        {tab.label}
                      </button>
                    ))}
                    <button onClick={() => setTrailOpen(true)}
                            className="flex-1 py-1.5 rounded-lg transition-all"
                            style={{
                              background: 'transparent', color: '#64748b', cursor: 'pointer',
                              fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 1.5,
                              border: '1px solid transparent',
                            }}
                            onMouseEnter={e => { const el = e.currentTarget; el.style.color = '#a78bfa'; el.style.background = 'rgba(167,139,250,0.1)' }}
                            onMouseLeave={e => { const el = e.currentTarget; el.style.color = '#64748b'; el.style.background = 'transparent' }}>
                      3D Trail ↗
                    </button>
                  </div>
                </div>

                {/* Tab content */}
                <div className="px-4 pb-4 space-y-3">
                  {activeTab === 'fusion' && <SensorFusion track={selected} />}
                  {activeTab === 'xai' && selected.xai && (
                    <XAIPanel xai={selected.xai} aiClass={selected.ai_class} />
                  )}
                  <ExpertApproval
                    track={selected}
                    approval={approvals[selected.track_id] ?? null}
                    onDecide={handleDecide}
                  />
                </div>
              </div>
            </div>
          )
        })()}
      </div>

      {/* 3D Trail Modal */}
      {trailOpen && selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center"
             style={{ background: 'rgba(0,0,0,0.82)', backdropFilter: 'blur(6px)' }}
             onClick={() => setTrailOpen(false)}>
          <div onClick={e => e.stopPropagation()} style={{
            background: 'linear-gradient(135deg,rgba(8,14,26,0.99) 0%,rgba(14,22,40,0.98) 100%)',
            border:     `1px solid ${CLASS_STYLES[selected.ai_class]?.color ?? '#38bdf8'}40`,
            borderTop:  `2px solid ${CLASS_STYLES[selected.ai_class]?.color ?? '#38bdf8'}`,
            borderRadius: 16, width: '90vw', maxWidth: 980,
            maxHeight: '88vh', overflow: 'hidden',
            boxShadow: '0 32px 100px rgba(0,0,0,0.85)',
          }}>
            <div className="flex items-center justify-between px-6 py-3"
                 style={{ borderBottom: '1px solid rgba(255,255,255,0.06)' }}>
              <div>
                <span style={{ fontFamily: 'Orbitron, monospace', color: '#a78bfa', fontSize: 11, letterSpacing: 3, fontWeight: 700 }}>3D MANEUVER ENVELOPE</span>
                <span style={{ color: '#475569', fontSize: 12, marginLeft: 12 }}>{selected.track_id} · {selected.ai_class}</span>
              </div>
              <button onClick={() => setTrailOpen(false)}
                      style={{ background: 'none', border: 'none', color: '#64748b', fontSize: 20, cursor: 'pointer' }}>✕</button>
            </div>
            <div style={{ padding: '8px 16px 16px' }}>
              <Trail3D track={selected} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
