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

  useEffect(() => {
    api.getTracks()
      .then(data => { setTracks(data); setSelectedId(data[0]?.track_id ?? null) })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  // Reset tab on track change
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
      <Header pendingCount={pendingCount} approvedCount={approvedCount} />

      <div className="flex flex-1 overflow-hidden">
        {/* Map — 60% */}
        <div className="p-3" style={{ width: '60%', minWidth: 0 }}>
          <TacticalMap tracks={tracks} selectedId={selectedId} onSelect={setSelectedId} />
        </div>

        {/* Side panel — 40% */}
        {selected && (() => {
          const approval  = approvals[selected.track_id]
          const dispCls   = approval?.action === 'override' ? (approval.override_class ?? selected.ai_class) : selected.ai_class
          const dispStyle = CLASS_STYLES[dispCls] ?? CLASS_STYLES['NEUTRAL']

          return (
            <div
              className="flex flex-col overflow-hidden"
              style={{ width: '40%', borderLeft: '1px solid rgba(56,189,248,0.1)', background: 'rgba(6,10,16,0.92)' }}
            >
              {/* Classification badge */}
              <div className="px-3 pt-3 pb-2 flex-shrink-0">
                <div
                  className="rounded-xl px-4 py-3 result-reveal"
                  style={{
                    background: `linear-gradient(135deg,${dispStyle.bg} 0%,rgba(0,0,0,0.45) 100%)`,
                    border:     `1px solid ${dispStyle.color}44`,
                    boxShadow:  `0 0 30px ${dispStyle.color}14`,
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-xs tracking-widest mb-0.5"
                         style={{ color: '#475569', fontFamily: 'Orbitron, monospace', fontSize: 9 }}>
                        AI CLASSIFICATION
                      </p>
                      <p className="font-black text-lg tracking-widest"
                         style={{ color: dispStyle.color, fontFamily: 'Orbitron, monospace', letterSpacing: 4 }}>
                        {dispStyle.icon} {dispCls}
                      </p>
                      <p className="text-xs mt-0.5" style={{ color: '#64748b' }}>
                        {(selected.ai_conf * 100).toFixed(1)}% confidence &nbsp;·&nbsp; {selected.track_id}
                      </p>
                    </div>
                    <div className="text-right text-xs space-y-0.5" style={{ color: '#94a3b8', lineHeight: 1.9 }}>
                      <div>{selected.electronic_signature}</div>
                      <div>{selected.flight_profile}</div>
                      <div>{selected.altitude_ft.toLocaleString()} ft</div>
                      <div>{selected.speed_kts.toFixed(0)} kts · {selected.rcs_m2.toFixed(1)} m²</div>
                      <div>{selected.weather} · {selected.thermal_signature}</div>
                    </div>
                  </div>
                  {approval && (
                    <p className="text-xs mt-2" style={{ color: approval.action === 'approved' ? '#4ade80' : '#fbbf24' }}>
                      {approval.action === 'approved' ? '✓ Expert approved' : `↺ Overridden → ${approval.override_class}`}
                    </p>
                  )}
                </div>
              </div>

              {/* Anomalies */}
              {selected.anomalies.length > 0 && (
                <div className="px-3 pb-2 flex-shrink-0">
                  <AnomalyAlerts anomalies={selected.anomalies} />
                </div>
              )}

              {/* Tab bar */}
              <div className="px-3 pb-2 flex-shrink-0">
                <div className="flex gap-1 p-1 rounded-xl" style={{ background: 'rgba(12,18,30,0.8)', border: '1px solid rgba(56,189,248,0.1)' }}>
                  {TABS.map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className="flex-1 py-1.5 rounded-lg text-xs transition-all"
                      style={{
                        background:  activeTab === tab.id ? 'rgba(56,189,248,0.18)' : 'transparent',
                        color:       activeTab === tab.id ? '#38bdf8' : '#475569',
                        fontFamily:  'Space Grotesk, sans-serif',
                        fontWeight:  activeTab === tab.id ? 600 : 400,
                        border:      activeTab === tab.id ? '1px solid rgba(56,189,248,0.3)' : '1px solid transparent',
                      }}
                    >
                      {tab.icon} {tab.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Tab content */}
              <div className="flex-1 overflow-y-auto px-3 pb-3 space-y-3 min-h-0">
                {activeTab === 'fusion' && <SensorFusion track={selected} />}

                {activeTab === 'xai' && selected.xai && (
                  <XAIPanel xai={selected.xai} aiClass={selected.ai_class} />
                )}

                {activeTab === 'trail' && <Trail3D track={selected} />}

                {/* Expert approval + What-If always visible below tab content */}
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
