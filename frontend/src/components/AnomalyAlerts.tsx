import type { Anomaly } from '../types'

interface Props { anomalies: Anomaly[] }

export default function AnomalyAlerts({ anomalies }: Props) {
  if (anomalies.length === 0) {
    return (
      <div className="rounded-xl px-3 py-2.5 text-xs flex items-center gap-2" style={{ background: 'rgba(34,197,94,0.06)', border: '1px solid rgba(34,197,94,0.2)', color: '#4ade80' }}>
        <span>✓</span> No anomalies detected
      </div>
    )
  }

  return (
    <div className="space-y-2">
      <p className="text-xs font-semibold tracking-widest uppercase" style={{ color: '#f59e0b88', fontFamily: 'Orbitron, monospace' }}>
        Anomaly Detection
      </p>
      {anomalies.map((a, i) => (
        <div
          key={i}
          className="rounded-xl px-3 py-2.5"
          style={{ background: 'rgba(245,158,11,0.07)', border: '1px solid rgba(245,158,11,0.28)', borderLeft: '3px solid #f59e0b' }}
        >
          <p style={{ color: '#fbbf24', fontSize: 13, fontWeight: 700, marginBottom: 3 }}>⚠ {a.title}</p>
          <p style={{ color: '#d97706', fontSize: 12 }}>{a.desc}</p>
        </div>
      ))}
    </div>
  )
}
