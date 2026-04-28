import type { Track } from '../types'
import { CLASS_STYLES } from '../types'

interface Props { track: Track }

const QUALITY_CONFIG = {
  HIGH:   { color: '#4ade80', dots: 3, label: 'HIGH CONFIDENCE' },
  MEDIUM: { color: '#fbbf24', dots: 2, label: 'MEDIUM CONFIDENCE' },
  LOW:    { color: '#f87171', dots: 1, label: 'LOW CONFIDENCE' },
}

const UNCERTAINTY_CONFIG = {
  LOW:    { color: '#4ade80', label: 'LOW — model certain' },
  MEDIUM: { color: '#fbbf24', label: 'MEDIUM — sensor ambiguity' },
  HIGH:   { color: '#f87171', label: 'HIGH — epistemic gap' },
}

function QualityDots({ count }: { count: number }) {
  return (
    <div className="flex gap-1">
      {[1, 2, 3].map(i => (
        <div
          key={i}
          style={{
            width: 8, height: 8, borderRadius: '50%',
            background: i <= count
              ? (count === 3 ? '#4ade80' : count === 2 ? '#fbbf24' : '#f87171')
              : 'rgba(255,255,255,0.08)',
          }}
        />
      ))}
    </div>
  )
}

function MiniBar({ value, color, height = 5 }: { value: number; color: string; height?: number }) {
  return (
    <div style={{ flex: 1, height, borderRadius: 3, background: 'rgba(255,255,255,0.07)', overflow: 'hidden' }}>
      <div style={{
        width: `${Math.min(100, value * 100)}%`, height: '100%',
        background: color, borderRadius: 3,
        transition: 'width 0.6s ease',
      }} />
    </div>
  )
}

export default function TrackInsights({ track }: Props) {
  const { track_quality: tq, epistemic_uncertainty: eu, uncertainty_label: ul, ds_fusion: ds, fusion } = track

  const qCfg = tq ? QUALITY_CONFIG[tq.label] : null
  const uCfg = ul  ? UNCERTAINTY_CONFIG[ul]   : null
  const dsAgreesWithWeighted = ds?.best === fusion.best

  return (
    <div
      className="rounded-xl px-4 py-3 space-y-3"
      style={{ background: 'rgba(10,16,28,0.9)', border: '1px solid rgba(56,189,248,0.18)' }}
    >
      <p style={{ color: '#7dd3fc', fontFamily: 'Orbitron, monospace', fontSize: 10, letterSpacing: 3, textTransform: 'uppercase', fontWeight: 700 }}>
        Track Intelligence
      </p>

      {/* Row 1: Kalman track quality + Epistemic uncertainty */}
      <div className="grid grid-cols-2 gap-3">

        {/* Kalman Track Quality */}
        {tq && qCfg ? (
          <div
            className="rounded-lg px-3 py-2.5 space-y-2"
            style={{ background: 'rgba(6,10,16,0.7)', border: `1px solid ${qCfg.color}33` }}
          >
            <div className="flex items-center justify-between">
              <span style={{ color: '#94a3b8', fontSize: 10, fontFamily: 'Orbitron, monospace', letterSpacing: 2 }}>KALMAN TQ</span>
              <QualityDots count={qCfg.dots} />
            </div>
            <p style={{ color: qCfg.color, fontFamily: 'Orbitron, monospace', fontSize: 13, fontWeight: 700, letterSpacing: 1 }}>
              {tq.label}
            </p>
            <div className="flex items-center gap-2">
              <MiniBar value={1 - tq.uncertainty} color={qCfg.color} />
              <span style={{ color: qCfg.color, fontSize: 11, fontWeight: 600, minWidth: 28 }}>
                {((1 - tq.uncertainty) * 100).toFixed(0)}%
              </span>
            </div>
            <p style={{ color: '#64748b', fontSize: 10 }}>
              σ² = {tq.covariance_trace.toFixed(4)}
            </p>
          </div>
        ) : (
          <div className="rounded-lg px-3 py-2.5" style={{ background: 'rgba(6,10,16,0.5)', border: '1px solid rgba(255,255,255,0.06)' }}>
            <p style={{ color: '#334155', fontSize: 11 }}>Track quality N/A</p>
          </div>
        )}

        {/* Epistemic Uncertainty */}
        {eu !== undefined && uCfg ? (
          <div
            className="rounded-lg px-3 py-2.5 space-y-2"
            style={{ background: 'rgba(6,10,16,0.7)', border: `1px solid ${uCfg.color}33` }}
          >
            <div className="flex items-center justify-between">
              <span style={{ color: '#94a3b8', fontSize: 10, fontFamily: 'Orbitron, monospace', letterSpacing: 2 }}>EPISTEMIC</span>
              <span style={{ color: uCfg.color, fontSize: 10, fontWeight: 700, fontFamily: 'Orbitron, monospace' }}>{ul}</span>
            </div>
            <p style={{ color: uCfg.color, fontFamily: 'Orbitron, monospace', fontSize: 13, fontWeight: 700 }}>
              {(eu * 100).toFixed(1)}%
            </p>
            <div className="flex items-center gap-2">
              <MiniBar value={eu} color={uCfg.color} />
            </div>
            <p style={{ color: '#64748b', fontSize: 10 }}>MC Dropout · 20 samples</p>
          </div>
        ) : null}
      </div>

      {/* Kalman smoothed position */}
      {tq && (
        <div className="flex items-center gap-3 px-1">
          <span style={{ color: '#475569', fontSize: 10, fontFamily: 'Orbitron, monospace', letterSpacing: 2, flexShrink: 0 }}>KF POS</span>
          <span style={{ color: '#94a3b8', fontSize: 11 }}>
            {tq.smoothed_position.lat.toFixed(4)}°N &nbsp;
            {tq.smoothed_position.lon.toFixed(4)}°E &nbsp;·&nbsp;
            {tq.smoothed_position.alt_ft.toLocaleString()} ft
          </span>
        </div>
      )}

      {/* DS Fusion agreement / divergence */}
      {ds && (
        <div
          className="rounded-lg px-3 py-2.5"
          style={{
            background: 'rgba(6,10,16,0.7)',
            border: `1px solid ${dsAgreesWithWeighted ? 'rgba(74,222,128,0.2)' : 'rgba(251,191,36,0.25)'}`,
          }}
        >
          <div className="flex items-center justify-between mb-2">
            <span style={{ color: '#94a3b8', fontSize: 10, fontFamily: 'Orbitron, monospace', letterSpacing: 2 }}>
              DEMPSTER-SHAFER
            </span>
            <span style={{
              color: ds.conflict_mass > 0.6 ? '#f87171' : ds.conflict_mass > 0.3 ? '#fbbf24' : '#4ade80',
              fontSize: 10, fontFamily: 'Orbitron, monospace',
            }}>
              CONFLICT {(ds.conflict_mass * 100).toFixed(0)}%
            </span>
          </div>

          <div className="flex items-center gap-2">
            {/* DS best class */}
            <span style={{
              color: CLASS_STYLES[ds.best]?.color ?? '#94a3b8',
              fontFamily: 'Orbitron, monospace', fontSize: 12, fontWeight: 700, letterSpacing: 1,
            }}>
              {CLASS_STYLES[ds.best]?.icon} {ds.best}
            </span>
            <span style={{ color: '#475569', fontSize: 11 }}>via DS</span>

            {!dsAgreesWithWeighted && (
              <span style={{
                marginLeft: 'auto',
                color: '#fbbf24', fontSize: 10, fontWeight: 600,
                background: 'rgba(251,191,36,0.12)', padding: '1px 8px', borderRadius: 4,
                border: '1px solid rgba(251,191,36,0.25)',
              }}>
                ≠ Weighted ({fusion.best})
              </span>
            )}
            {dsAgreesWithWeighted && (
              <span style={{
                marginLeft: 'auto',
                color: '#4ade80', fontSize: 10,
                background: 'rgba(74,222,128,0.08)', padding: '1px 8px', borderRadius: 4,
                border: '1px solid rgba(74,222,128,0.2)',
              }}>
                ✓ Agrees
              </span>
            )}
          </div>

          {/* Top DS probabilities */}
          <div className="mt-2 space-y-1">
            {Object.entries(ds.probs)
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([cls, prob]) => {
                const s = CLASS_STYLES[cls] ?? CLASS_STYLES['NEUTRAL']
                return (
                  <div key={cls} className="flex items-center gap-2">
                    <span style={{ color: '#64748b', fontSize: 10, width: 96, flexShrink: 0 }}>{cls}</span>
                    <MiniBar value={prob} color={s.color} height={4} />
                    <span style={{ color: s.color, fontSize: 10, fontWeight: 600, minWidth: 32, textAlign: 'right' }}>
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                )
              })}
          </div>
        </div>
      )}
    </div>
  )
}
