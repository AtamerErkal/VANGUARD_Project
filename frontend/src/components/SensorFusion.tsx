import type { Track } from '../types'
import { CLASS_STYLES, SENSOR_ORDER } from '../types'

interface Props { track: Track }

function ConfBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-2 flex-1">
      <div className="flex-1 h-1 rounded-full" style={{ background: 'rgba(255,255,255,0.07)' }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${value * 100}%`, background: color }}
        />
      </div>
      <span className="text-xs font-bold w-9 text-right" style={{ color }}>{(value * 100).toFixed(0)}%</span>
    </div>
  )
}

export default function SensorFusion({ track }: Props) {
  const { sensor_votes, fusion } = track

  const sortedProbs = Object.entries(fusion.probs).sort((a, b) => b[1] - a[1])

  return (
    <div className="space-y-3">
      <p className="text-xs font-semibold tracking-widest uppercase" style={{ color: '#38bdf888', fontFamily: 'Orbitron, monospace' }}>
        Sensor Fusion
      </p>

      {/* Sensor cards — staggered slide-in via CSS */}
      <div className="space-y-2">
        {SENSOR_ORDER.map(key => {
          const vd    = sensor_votes[key]
          if (!vd) return null
          const style = CLASS_STYLES[vd.vote] ?? CLASS_STYLES['NEUTRAL']
          const w     = fusion.weights[key] ?? 0

          return (
            <div
              key={key}
              className="sensor-card rounded-xl px-3 py-2.5"
              style={{
                background:  'rgba(12,18,30,0.85)',
                border:      `1px solid ${style.color}33`,
                borderLeft:  `3px solid ${style.color}`,
              }}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-sm font-semibold text-slate-200">
                  {vd.icon} {vd.label}
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-xs px-1.5 py-0.5 rounded" style={{ background: style.bg, color: style.color, fontFamily: 'Orbitron, monospace', letterSpacing: 1, fontSize: 10 }}>
                    {vd.vote}
                  </span>
                  <span className="text-xs" style={{ color: '#475569' }}>w={w.toFixed(2)}</span>
                </div>
              </div>
              <p className="text-xs mb-2" style={{ color: '#64748b' }}>{vd.reading}</p>
              <ConfBar value={vd.conf} color={style.color} />
            </div>
          )
        })}
      </div>

      {/* Probability distribution */}
      <div className="rounded-xl px-3 py-3 space-y-2" style={{ background: 'rgba(12,18,30,0.7)', border: '1px solid rgba(56,189,248,0.1)' }}>
        <p className="text-xs tracking-widest uppercase mb-2" style={{ color: '#38bdf866', fontFamily: 'Orbitron, monospace', fontSize: 9 }}>
          Fused Probability
        </p>
        {sortedProbs.map(([cls, prob]) => {
          const s = CLASS_STYLES[cls] ?? CLASS_STYLES['NEUTRAL']
          return (
            <div key={cls} className="flex items-center gap-2">
              <span className="w-28 text-xs truncate" style={{ color: '#64748b' }}>{cls}</span>
              <div className="flex-1 h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.05)' }}>
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{ width: `${prob * 100}%`, background: s.color, opacity: cls === fusion.best ? 1 : 0.45 }}
                />
              </div>
              <span className="text-xs w-9 text-right" style={{ color: cls === fusion.best ? s.color : '#475569', fontWeight: cls === fusion.best ? 700 : 400 }}>
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
