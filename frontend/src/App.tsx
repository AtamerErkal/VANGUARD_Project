import { useEffect, useState } from 'react'
import type { Track, ApprovalState } from './types'
import { CLASS_STYLES } from './types'
import { api } from './api'
import Header       from './components/Header'
import TacticalMap  from './components/TacticalMap'
import SensorFusion from './components/SensorFusion'
import AnomalyAlerts from './components/AnomalyAlerts'
import Trail3D       from './components/Trail3D'
import ExpertApproval from './components/ExpertApproval'
import WhatIfPanel   from './components/WhatIfPanel'

export default function App() {
  const [tracks,     setTracks]     = useState<Track[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [approvals,  setApprovals]  = useState<Record<string, ApprovalState>>({})
  const [loading,    setLoading]    = useState(true)
  const [error,      setError]      = useState<string | null>(null)

  useEffect(() => {
    api.getTracks()
      .then(data => { setTracks(data); setSelectedId(data[0]?.track_id ?? null) })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [])

  const selected = tracks.find(t => t.track_id === selectedId) ?? null

  const pendingCount  = tracks.filter(t => !approvals[t.track_id]).length
  const approvedCount = Object.values(approvals).filter(a => a.action === 'approved').length

  const handleDecide = (id: string, state: ApprovalState) =>
    setApprovals(prev => ({ ...prev, [id]: state }))

  if (loading) return (
    <div className="h-screen flex items-center justify-center" style={{ background: '#060a10' }}>
      <div className="text-center space-y-3">
        <div className="text-2xl animate-pulse" style={{ fontFamily: 'Orbitron, monospace', color: '#38bdf8', letterSpacing: 4 }}>
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
        <p className="text-xs" style={{ color: '#334155' }}>Start the FastAPI server: <code className="text-sky-400">uvicorn backend.main:app --reload</code></p>
      </div>
    </div>
  )

  return (
    <div className="h-screen flex flex-col overflow-hidden" style={{ background: '#060a10' }}>
      <Header pendingCount={pendingCount} approvedCount={approvedCount} />

      <div className="flex flex-1 overflow-hidden gap-0">
        {/* Map — left */}
        <div className="flex-1 p-3">
          <TacticalMap tracks={tracks} selectedId={selectedId} onSelect={setSelectedId} />
        </div>

        {/* Side panel — right */}
        {selected && (
          <div
            className="w-96 flex flex-col overflow-y-auto p-3 gap-3"
            style={{ borderLeft: '1px solid rgba(56,189,248,0.1)', background: 'rgba(6,10,16,0.8)' }}
          >
            {/* Classification badge */}
            {(() => {
              const approval = approvals[selected.track_id]
              const dispCls  = approval?.action === 'override' ? (approval.override_class ?? selected.ai_class) : selected.ai_class
              const style    = CLASS_STYLES[dispCls] ?? CLASS_STYLES['NEUTRAL']
              return (
                <div
                  className="rounded-xl px-4 py-3 result-reveal"
                  style={{
                    background:  `linear-gradient(135deg, ${style.bg} 0%, rgba(0,0,0,0.4) 100%)`,
                    border:      `1px solid ${style.color}44`,
                    boxShadow:   `0 0 30px ${style.color}18`,
                  }}
                >
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="text-xs tracking-widest mb-1" style={{ color: '#475569', fontFamily: 'Orbitron, monospace', fontSize: 9 }}>
                        AI CLASSIFICATION
                      </p>
                      <p className="font-black text-lg tracking-widest" style={{ color: style.color, fontFamily: 'Orbitron, monospace', letterSpacing: 4 }}>
                        {style.icon} {dispCls}
                      </p>
                      <p className="text-xs mt-0.5" style={{ color: '#64748b' }}>
                        {(selected.ai_conf * 100).toFixed(1)}% confidence &nbsp;·&nbsp; {selected.track_id}
                      </p>
                    </div>
                    <div className="text-right text-xs space-y-0.5" style={{ color: '#475569', lineHeight: 1.8 }}>
                      <div>{selected.electronic_signature}</div>
                      <div>{selected.flight_profile}</div>
                      <div>{selected.altitude_ft.toLocaleString()} ft</div>
                      <div>{selected.speed_kts.toFixed(0)} kts · {selected.rcs_m2.toFixed(1)} m²</div>
                    </div>
                  </div>
                  {approval && (
                    <p className="text-xs mt-2" style={{ color: approval.action === 'approved' ? '#4ade80' : '#fbbf24' }}>
                      {approval.action === 'approved' ? '✓ Expert approved' : `↺ Overridden → ${approval.override_class}`}
                    </p>
                  )}
                </div>
              )
            })()}

            <AnomalyAlerts anomalies={selected.anomalies} />
            <SensorFusion  track={selected} />

            <div>
              <p className="text-xs font-semibold tracking-widest uppercase mb-2" style={{ color: '#38bdf888', fontFamily: 'Orbitron, monospace' }}>
                3D Flight Trail
              </p>
              <Trail3D track={selected} />
            </div>

            <ExpertApproval
              track={selected}
              approval={approvals[selected.track_id] ?? null}
              onDecide={handleDecide}
            />

            <WhatIfPanel track={selected} />
          </div>
        )}
      </div>
    </div>
  )
}
