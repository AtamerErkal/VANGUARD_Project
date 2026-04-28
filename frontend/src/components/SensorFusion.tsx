import { useState } from 'react'
import type { Track } from '../types'
import { CLASS_STYLES, SENSOR_ORDER } from '../types'

interface Props { track: Track }

const SENSOR_ICONS: Record<string, string> = {
  radar: '📡',
  esm:   '〰',
  irst:  '🔥',
  iff:   '🆔',
}

function ConfBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="flex items-center gap-2 flex-1">
      <div className="flex-1 h-2 rounded-full" style={{ background: 'rgba(255,255,255,0.08)' }}>
        <div className="h-full rounded-full transition-all duration-700"
             style={{ width: `${value * 100}%`, background: color }} />
      </div>
      <span style={{ color, fontWeight: 700, fontSize: 12, minWidth: 34, textAlign: 'right' }}>
        {(value * 100).toFixed(0)}%
      </span>
    </div>
  )
}

export default function SensorFusion({ track }: Props) {
  const { sensor_votes, fusion, ds_fusion } = track
  const [showDS, setShowDS] = useState(false)

  const activeProbs  = showDS && ds_fusion ? ds_fusion.probs  : fusion.probs
  const activeBest   = showDS && ds_fusion ? ds_fusion.best   : fusion.best
  const sortedProbs  = Object.entries(activeProbs).sort((a, b) => b[1] - a[1])

  return (
    <div className="space-y-3">
      <p style={{ color: '#7dd3fc', fontFamily: 'Orbitron, monospace', fontSize: 10, letterSpacing: 3, textTransform: 'uppercase', fontWeight: 700 }}>
        Sensor Fusion
      </p>

      {/* Sensor vote cards */}
      <div className="grid grid-cols-2 gap-2">
        {SENSOR_ORDER.map(key => {
          const vd    = sensor_votes[key]
          if (!vd) return null
          const style = CLASS_STYLES[vd.vote] ?? CLASS_STYLES['NEUTRAL']
          const w     = fusion.weights[key] ?? 0
          const icon  = SENSOR_ICONS[key] ?? '◉'

          return (
            <div key={key} className="rounded-xl px-3 py-3"
                 style={{
                   background:  'rgba(10,16,28,0.9)',
                   border:      `1px solid ${style.color}55`,
                   borderLeft:  `3px solid ${style.color}`,
                   boxShadow:   `inset 0 0 16px ${style.color}08`,
                 }}>
              <div className="flex items-center justify-between mb-2">
                <span style={{ color: '#e2e8f0', fontWeight: 700, fontSize: 13 }}>
                  {icon} {vd.label}
                </span>
                <span style={{ color: '#64748b', fontSize: 11 }}>w={w.toFixed(2)}</span>
              </div>
              <div className="flex items-center justify-between mb-2">
                <p style={{ color: '#cbd5e1', fontSize: 11 }}>{vd.reading}</p>
                <span className="px-2 py-0.5 rounded ml-2 flex-shrink-0"
                      style={{ background: style.bg, color: style.color, fontFamily: 'Orbitron, monospace', fontSize: 10, fontWeight: 800, letterSpacing: 0.5, border: `1px solid ${style.color}44` }}>
                  {style.icon} {style.nato}
                </span>
              </div>
              <ConfBar value={vd.conf} color={style.color} />
            </div>
          )
        })}
      </div>

      {/* Fusion method toggle */}
      {ds_fusion && (
        <div className="flex gap-1 p-1 rounded-lg" style={{ background: 'rgba(6,10,16,0.8)', border: '1px solid rgba(56,189,248,0.12)' }}>
          <button
            onClick={() => setShowDS(false)}
            className="flex-1 py-1.5 rounded-md text-xs transition-all"
            style={{
              background: !showDS ? 'rgba(56,189,248,0.18)' : 'transparent',
              color:      !showDS ? '#7dd3fc' : '#475569',
              fontFamily: 'Orbitron, monospace', fontSize: 10, letterSpacing: 1,
              border:     !showDS ? '1px solid rgba(56,189,248,0.3)' : '1px solid transparent',
              fontWeight: !showDS ? 700 : 400,
            }}
          >
            Weighted
          </button>
          <button
            onClick={() => setShowDS(true)}
            className="flex-1 py-1.5 rounded-md text-xs transition-all"
            style={{
              background: showDS ? 'rgba(251,191,36,0.15)' : 'transparent',
              color:      showDS ? '#fbbf24' : '#475569',
              fontFamily: 'Orbitron, monospace', fontSize: 10, letterSpacing: 1,
              border:     showDS ? '1px solid rgba(251,191,36,0.3)' : '1px solid transparent',
              fontWeight: showDS ? 700 : 400,
            }}
          >
            Dempster-Shafer
          </button>
        </div>
      )}

      {/* Probability distribution */}
      <div className="rounded-xl px-3 py-3 space-y-2"
           style={{ background: 'rgba(10,16,28,0.85)', border: `1px solid ${showDS ? 'rgba(251,191,36,0.2)' : 'rgba(56,189,248,0.15)'}` }}>
        <div className="flex items-center justify-between mb-2">
          <p style={{ color: showDS ? '#fbbf24aa' : '#38bdf899', fontFamily: 'Orbitron, monospace', fontSize: 10, letterSpacing: 3, textTransform: 'uppercase' }}>
            {showDS ? 'DS Evidence' : 'Fused Probability'}
          </p>
          {showDS && ds_fusion && (
            <span style={{ color: ds_fusion.conflict_mass > 0.6 ? '#f87171' : '#fbbf24', fontSize: 10, fontFamily: 'Orbitron, monospace' }}>
              K={( ds_fusion.conflict_mass * 100).toFixed(0)}%
            </span>
          )}
        </div>
        {sortedProbs.map(([cls, prob]) => {
          const s       = CLASS_STYLES[cls] ?? CLASS_STYLES['NEUTRAL']
          const isBest  = cls === activeBest
          return (
            <div key={cls} className="flex items-center gap-2">
              <div className="flex items-center gap-1.5" style={{ width: 120, flexShrink: 0 }}>
                <span style={{ color: s.color, fontSize: 11 }}>{s.icon}</span>
                <span style={{ color: isBest ? '#e2e8f0' : '#94a3b8', fontSize: 12, fontWeight: isBest ? 700 : 400 }}>
                  {cls}
                </span>
              </div>
              <div className="flex-1 h-2 rounded-full" style={{ background: 'rgba(255,255,255,0.07)' }}>
                <div className="h-full rounded-full transition-all duration-700"
                     style={{ width: `${prob * 100}%`, background: s.color, opacity: isBest ? 1 : 0.5 }} />
              </div>
              <span style={{ color: isBest ? s.color : '#64748b', fontSize: 12, fontWeight: isBest ? 700 : 400, minWidth: 38, textAlign: 'right' }}>
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

