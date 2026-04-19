import type { Track } from '../types'
import { CLASS_STYLES, SENSOR_ORDER } from '../types'

interface Props { track: Track }

function ConfBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-2 flex-1">
      <div className="flex-1 h-2 rounded-full" style={{ background: 'rgba(255,255,255,0.07)' }}>
        <div className="h-full rounded-full transition-all duration-700"
             style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span style={{ color, fontWeight: 700, fontSize: 13, minWidth: 36, textAlign: 'right' }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  )
}

export default function SensorFusion({ track }: Props) {
  const { sensor_votes, fusion } = track
  const sortedProbs = Object.entries(fusion.probs).sort((a, b) => b[1] - a[1])

  return (
    <div className="space-y-3">
      <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 11, letterSpacing: 3, textTransform: 'uppercase', fontWeight: 600 }}>
        Sensor Fusion
      </p>

      <div className="grid grid-cols-2 gap-2">
        {SENSOR_ORDER.map(key => {
          const vd    = sensor_votes[key]
          if (!vd) return null
          const style = CLASS_STYLES[vd.vote] ?? CLASS_STYLES['NEUTRAL']
          const w     = fusion.weights[key] ?? 0

          return (
            <div key={key} className="sensor-card rounded-xl px-3 py-3"
                 style={{ background: 'rgba(12,18,30,0.85)', border: `1px solid ${style.color}44`, borderLeft: `3px solid ${style.color}` }}>
              <div className="flex items-center justify-between mb-2">
                <span style={{ color: '#f1f5f9', fontWeight: 600, fontSize: 14 }}>
                  {vd.icon} {vd.label}
                </span>
                <div className="flex items-center gap-2">
                  <span className="px-2 py-0.5 rounded"
                        style={{ background: style.bg, color: style.color, fontFamily: 'Orbitron, monospace', letterSpacing: 1, fontSize: 11, fontWeight: 700 }}>
                    {vd.vote}
                  </span>
                  <span style={{ color: '#94a3b8', fontSize: 12 }}>w={w.toFixed(2)}</span>
                </div>
              </div>
              <p className="mb-2" style={{ color: '#cbd5e1', fontSize: 12 }}>{vd.reading}</p>
              <ConfBar value={vd.conf} color={style.color} />
            </div>
          )
        })}
      </div>

      {/* Probability distribution */}
      <div className="rounded-xl px-3 py-3 space-y-2"
           style={{ background: 'rgba(12,18,30,0.7)', border: '1px solid rgba(56,189,248,0.12)' }}>
        <p style={{ color: '#38bdf899', fontFamily: 'Orbitron, monospace', fontSize: 10, letterSpacing: 3, textTransform: 'uppercase', marginBottom: 8 }}>
          Fused Probability
        </p>
        {sortedProbs.map(([cls, prob]) => {
          const s       = CLASS_STYLES[cls] ?? CLASS_STYLES['NEUTRAL']
          const isBest  = cls === fusion.best
          return (
            <div key={cls} className="flex items-center gap-2">
              <span style={{ color: isBest ? '#e2e8f0' : '#94a3b8', fontSize: 13, width: 112, flexShrink: 0, fontWeight: isBest ? 600 : 400 }}>
                {cls}
              </span>
              <div className="flex-1 h-2 rounded-full" style={{ background: 'rgba(255,255,255,0.06)' }}>
                <div className="h-full rounded-full transition-all duration-700"
                     style={{ width: `${prob * 100}%`, background: s.color, opacity: isBest ? 1 : 0.45 }} />
              </div>
              <span style={{ color: isBest ? s.color : '#64748b', fontSize: 13, fontWeight: isBest ? 700 : 400, minWidth: 40, textAlign: 'right' }}>
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
