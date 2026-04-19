import { useEffect, useState } from 'react'
import type { Track, ApprovalState } from './types'
import { CLASS_STYLES } from './types'
import { api } from './api'
import Header        from './components/Header'
import TacticalMap   from './components/TacticalMap'
import SensorFusion  from './components/SensorFusion'
import AnomalyAlerts from './components/AnomalyAlerts'
import Trail3D       from './components/Trail3D'
import ExpertApproval from './components/ExpertApproval'
import WhatIfPanel   from './components/WhatIfPanel'
import XAIPanel      from './components/XAIPanel'

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
      <Header pendingCount={pendingCount} approvedCount={approvedCount} selected={selected} />

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
              <div className="px-4 pt-4 pb-3 flex-shrink-0">
                <div
                  className="rounded-xl px-5 py-4 result-reveal"
                  style={{
                    background: `linear-gradient(135deg,${dispStyle.bg} 0%,rgba(0,0,0,0.45) 100%)`,
                    border:     `1px solid ${dispStyle.color}55`,
                    boxShadow:  `0 0 40px ${dispStyle.color}18`,
                  }}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p style={{ color: '#94a3b8', fontFamily: 'Orbitron, monospace', fontSize: 10, letterSpacing: 3, marginBottom: 6 }}>
                        AI CLASSIFICATION
                      </p>
                      <p style={{ color: dispStyle.color, fontFamily: 'Orbitron, monospace', fontSize: 22, fontWeight: 900, letterSpacing: 4, lineHeight: 1.1 }}>
                        {dispStyle.icon} {dispCls}
                      </p>
                      <p style={{ color: '#cbd5e1', fontSize: 13, marginTop: 6, fontWeight: 500 }}>
                        <span style={{ color: dispStyle.color, fontWeight: 700 }}>{(selected.ai_conf * 100).toFixed(1)}%</span>
                        {' '}confidence &nbsp;·&nbsp;
                        <span style={{ color: '#94a3b8' }}>{selected.track_id}</span>
                      </p>
                    </div>
                    <div className="text-right space-y-1 flex-shrink-0" style={{ color: '#cbd5e1', fontSize: 12, lineHeight: 1.8 }}>
                      <div style={{ color: '#94a3b8' }}>{selected.electronic_signature}</div>
                      <div style={{ color: '#94a3b8' }}>{selected.flight_profile}</div>
                      <div><span style={{ color: '#64748b', fontSize: 11 }}>Alt </span>{selected.altitude_ft.toLocaleString()} ft</div>
                      <div><span style={{ color: '#64748b', fontSize: 11 }}>Spd </span>{selected.speed_kts.toFixed(0)} kts &nbsp;<span style={{ color: '#64748b', fontSize: 11 }}>RCS </span>{selected.rcs_m2.toFixed(1)} m²</div>
                      <div style={{ color: '#94a3b8' }}>{selected.weather} · {selected.thermal_signature}</div>
                    </div>
                  </div>
                  {approval && (
                    <p style={{ color: approval.action === 'approved' ? '#4ade80' : '#fbbf24', fontSize: 12, marginTop: 8, fontWeight: 600 }}>
                      {approval.action === 'approved' ? '✓ Expert approved' : `↺ Overridden → ${approval.override_class}`}
                    </p>
                  )}
                </div>
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
