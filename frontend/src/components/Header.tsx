import { useEffect, useState } from 'react'
import type { Track, ModelStats } from '../types'
import { CLASS_STYLES } from '../types'
import { api } from '../api'

interface Props {
  pendingCount:  number
  approvedCount: number
  selected:      Track | null
  threatScore:   number
}

// ── Theater threat level ──────────────────────────────────────────────────────
const THREAT_LEVELS = [
  { min: 60, label: 'CRITICAL',  color: '#ef4444', bg: 'rgba(239,68,68,0.12)',   pulse: true  },
  { min: 35, label: 'HIGH',      color: '#f97316', bg: 'rgba(249,115,22,0.10)',  pulse: true  },
  { min: 18, label: 'ELEVATED',  color: '#f59e0b', bg: 'rgba(245,158,11,0.09)',  pulse: false },
  { min:  8, label: 'GUARDED',   color: '#eab308', bg: 'rgba(234,179,8,0.08)',   pulse: false },
  { min:  0, label: 'LOW',       color: '#22c55e', bg: 'rgba(34,197,94,0.07)',   pulse: false },
] as const

function getThreatLevel(score: number) {
  return THREAT_LEVELS.find(l => score >= l.min) ?? THREAT_LEVELS[THREAT_LEVELS.length - 1]
}

// ── Per-class colors ──────────────────────────────────────────────────────────
const CLS_COLOR: Record<string, string> = {
  HOSTILE:          '#ef4444',
  SUSPECT:          '#f59e0b',
  FRIEND:           '#22c55e',
  'ASSUMED FRIEND': '#22c55e',
  NEUTRAL:          '#94a3b8',
  CIVILIAN:         '#38bdf8',
}

const MODEL_FEATURES = [
  { group: 'Kinematic',     color: '#a78bfa', items: ['Altitude (ft)', 'Speed (kts)', 'Heading (°)', 'Latitude', 'Longitude'] },
  { group: 'Radar',         color: '#fb923c', items: ['Radar Cross-Section (m²)'] },
  { group: 'Electronic',    color: '#38bdf8', items: ['Electronic Signature / IFF'] },
  { group: 'Thermal',       color: '#f472b6', items: ['Thermal Signature (IRST)'] },
  { group: 'Environmental', color: '#4ade80', items: ['Weather condition', 'Flight Profile'] },
]

export default function Header({ pendingCount, approvedCount, selected, threatScore }: Props) {
  const [clock,       setClock]       = useState('')
  const [modelOpen,   setModelOpen]   = useState(false)
  const [sensorsOpen, setSensorsOpen] = useState(false)
  const [stats,       setStats]       = useState<ModelStats | null>(null)
  const [statsLoading,setStatsLoading]= useState(false)

  useEffect(() => {
    const tick = () => {
      const n = new Date(), pad = (x: number) => String(x).padStart(2, '0')
      setClock(`${pad(n.getUTCHours())}:${pad(n.getUTCMinutes())}:${pad(n.getUTCSeconds())} UTC`)
    }
    tick(); const id = setInterval(tick, 1000); return () => clearInterval(id)
  }, [])

  const openModel = () => {
    setModelOpen(true)
    if (!stats && !statsLoading) {
      setStatsLoading(true)
      api.getModelStats().then(s => { setStats(s); setStatsLoading(false) }).catch(() => setStatsLoading(false))
    }
  }

  const threat      = getThreatLevel(threatScore)
  const aiConf      = selected ? (selected.ai_conf * 100).toFixed(1) : '—'
  const aiClass     = selected?.ai_class ?? '—'
  const fusedProb   = selected ? (((selected.fusion?.probs?.[selected.ai_class]) ?? 0) * 100).toFixed(1) : '—'
  const fusedBest   = selected?.fusion?.best ?? '—'
  const accentStyle = selected ? (CLASS_STYLES[aiClass] ?? CLASS_STYLES['NEUTRAL']) : null
  const gap         = selected
    ? Math.abs(selected.ai_conf * 100 - ((selected.fusion?.probs?.[selected.ai_class] ?? 0) * 100)).toFixed(1)
    : '—'

  // Confusion matrix max (for color scaling)
  const cmMax = stats
    ? Math.max(...stats.confusion_matrix.flatMap(r => r))
    : 1

  return (
    <>
      <header
        className="relative flex items-center justify-between px-6 py-3 overflow-hidden flex-shrink-0"
        style={{
          background:   'linear-gradient(135deg, rgba(15,23,42,0.97) 0%, rgba(30,41,59,0.9) 100%)',
          borderBottom: '1px solid rgba(56,189,248,0.18)',
          boxShadow:    '0 0 40px rgba(56,189,248,0.06)',
        }}
      >
        <div className="absolute top-0 left-0 right-0 h-px"
             style={{ background: 'linear-gradient(90deg, transparent, #38bdf8, transparent)', animation: 'scan 3s ease-in-out infinite' }} />

        <div>
          <h1 style={{ fontFamily: 'Orbitron, monospace', color: '#38bdf8', letterSpacing: 6, fontSize: 18, fontWeight: 900 }}>
            🛡️ VANGUARD TACTICAL
          </h1>
          <p style={{ color: 'rgba(56,189,248,0.45)', letterSpacing: 3, fontFamily: 'Space Grotesk, sans-serif', fontSize: 12, marginTop: 2 }}>
            SENSOR FUSION &amp; THREAT ASSESSMENT SYSTEM
          </p>
        </div>

        <div className="flex items-center gap-5">
          {/* Theater threat level */}
          <div className="flex items-center gap-2 rounded-lg px-3 py-1.5"
               style={{ background: threat.bg, border: `1px solid ${threat.color}44` }}>
            {threat.pulse && (
              <span className="w-2 h-2 rounded-full animate-ping"
                    style={{ background: threat.color, opacity: 0.7, flexShrink: 0 }} />
            )}
            <span style={{ color: '#64748b', fontFamily: 'Orbitron, monospace', fontSize: 8, letterSpacing: 2 }}>THREAT</span>
            <span style={{ color: threat.color, fontFamily: 'Orbitron, monospace', fontSize: 11, fontWeight: 700, letterSpacing: 1 }}>
              {threat.label}
            </span>
            <span style={{ color: threat.color, fontFamily: 'Orbitron, monospace', fontSize: 10 }}>
              {threatScore}
            </span>
          </div>

          {/* Status buttons */}
          <div className="flex items-center gap-4 text-xs" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
            <button onClick={openModel}
                    className="flex items-center gap-1.5 hover:opacity-75 transition-opacity"
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
              <span className="w-2 h-2 rounded-full animate-[pulse-dot_2.2s_ease-in-out_infinite]"
                    style={{ background: '#22c55e', boxShadow: '0 0 6px #22c55e' }} />
              <span style={{ color: '#22c55e', fontWeight: 600 }}>Model Online</span>
            </button>
            <button onClick={() => setSensorsOpen(true)}
                    className="flex items-center gap-1.5 hover:opacity-75 transition-opacity"
                    style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}>
              <span className="w-2 h-2 rounded-full animate-[pulse-dot_2.2s_ease-in-out_infinite_0.4s]"
                    style={{ background: '#22c55e', boxShadow: '0 0 6px #22c55e' }} />
              <span style={{ color: '#22c55e', fontWeight: 600 }}>Sensors Active</span>
            </button>
          </div>

          {/* Queue */}
          <div style={{ fontFamily: 'Orbitron, monospace', fontSize: 12 }}>
            <span style={{ color: '#f59e0b' }}>{pendingCount}</span>
            <span style={{ color: '#475569', fontFamily: 'Space Grotesk' }}> pending &nbsp;</span>
            <span style={{ color: '#22c55e' }}>{approvedCount}</span>
            <span style={{ color: '#475569', fontFamily: 'Space Grotesk' }}> approved</span>
          </div>

          {/* Clock */}
          <div style={{ fontFamily: 'Orbitron, monospace', fontSize: 11, color: 'rgba(56,189,248,0.5)', letterSpacing: 3 }}>
            {clock}
          </div>
        </div>
      </header>

      {/* ── MODEL INFO MODAL ─────────────────────────────────────────────────── */}
      {modelOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center"
             style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)' }}
             onClick={() => setModelOpen(false)}>
          <div className="relative overflow-y-auto"
               style={{
                 background:   'linear-gradient(135deg, rgba(8,14,26,0.99) 0%, rgba(14,22,40,0.98) 100%)',
                 border:       '1px solid rgba(56,189,248,0.25)', borderTop: '2px solid #38bdf8',
                 borderRadius: 14, padding: '24px 28px',
                 maxWidth: 620, width: '92vw', maxHeight: '88vh',
                 boxShadow: '0 24px 80px rgba(0,0,0,0.8)',
                 fontFamily: 'Space Grotesk, sans-serif',
               }}
               onClick={e => e.stopPropagation()}>
            <button onClick={() => setModelOpen(false)}
                    style={{ position: 'absolute', top: 14, right: 16, fontSize: 18, background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>✕</button>

            {/* Header */}
            <div className="mb-5">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full" style={{ background: '#22c55e', boxShadow: '0 0 8px #22c55e' }} />
                <span style={{ fontFamily: 'Orbitron, monospace', fontSize: 9, color: '#22c55e', letterSpacing: 3 }}>MODEL ONLINE</span>
              </div>
              <h2 style={{ fontFamily: 'Orbitron, monospace', fontSize: 16, color: '#38bdf8', letterSpacing: 3 }}>ImprovedAircraftClassifier</h2>
              <p style={{ color: '#64748b', fontSize: 12, marginTop: 3 }}>PyTorch · 3-layer MLP · 6-class threat classification</p>
            </div>

            {/* Architecture */}
            <div className="mb-5">
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 10 }}>ARCHITECTURE</p>
              <div className="flex items-center gap-1 flex-wrap">
                {[
                  { label: 'Input',    sub: '10 features',                    color: '#475569' },
                  { label: 'Dense 64', sub: 'BatchNorm · ReLU · Dropout 0.2', color: '#38bdf8' },
                  { label: 'Dense 32', sub: 'BatchNorm · ReLU · Dropout 0.2', color: '#38bdf8' },
                  { label: 'Output',   sub: '6 classes · Softmax',            color: '#22c55e' },
                ].map((layer, i, arr) => (
                  <div key={layer.label} className="flex items-center gap-1">
                    <div className="rounded-lg px-3 py-2 text-center"
                         style={{ background: 'rgba(255,255,255,0.04)', border: `1px solid ${layer.color}33`, minWidth: 90 }}>
                      <div style={{ color: layer.color, fontFamily: 'Orbitron, monospace', fontSize: 11, fontWeight: 700 }}>{layer.label}</div>
                      <div style={{ color: '#64748b', fontSize: 10, marginTop: 2 }}>{layer.sub}</div>
                    </div>
                    {i < arr.length - 1 && <span style={{ color: '#334155', fontSize: 16 }}>→</span>}
                  </div>
                ))}
              </div>
            </div>

            {/* Features */}
            <div className="mb-5">
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 10 }}>INPUT FEATURES (10)</p>
              <div className="space-y-1.5">
                {MODEL_FEATURES.map(({ group, color, items }) => (
                  <div key={group} className="flex items-start gap-2">
                    <span className="rounded px-1.5 py-0.5 flex-shrink-0"
                          style={{ background: `${color}18`, color, fontSize: 10, fontWeight: 600, minWidth: 92, textAlign: 'center' }}>
                      {group}
                    </span>
                    <span style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{items.join(' · ')}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* ── REAL METRICS ── */}
            <div className="mb-5">
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 10 }}>
                EVALUATION METRICS — TEST SET
              </p>

              {statsLoading && (
                <p style={{ color: '#475569', fontSize: 12 }} className="animate-pulse">Computing metrics…</p>
              )}

              {stats && (
                <>
                  {/* Overall scores */}
                  <div className="grid grid-cols-3 gap-2 mb-4">
                    {[
                      { label: 'ACCURACY',    value: (stats.accuracy   * 100).toFixed(1) + '%', color: '#38bdf8' },
                      { label: 'F1 MACRO',    value: (stats.f1_macro   * 100).toFixed(1) + '%', color: '#a78bfa' },
                      { label: 'F1 WEIGHTED', value: (stats.f1_weighted * 100).toFixed(1) + '%', color: '#fb923c' },
                    ].map(({ label, value, color }) => (
                      <div key={label} className="rounded-xl py-3 text-center"
                           style={{ background: `${color}0d`, border: `1px solid ${color}28` }}>
                        <div style={{ color: '#64748b', fontFamily: 'Orbitron, monospace', fontSize: 8, letterSpacing: 2, marginBottom: 4 }}>{label}</div>
                        <div style={{ color, fontFamily: 'Orbitron, monospace', fontSize: 22, fontWeight: 900 }}>{value}</div>
                      </div>
                    ))}
                  </div>
                  <p style={{ color: '#334155', fontSize: 11, marginBottom: 12 }}>
                    Train: {stats.train_size} samples &nbsp;·&nbsp; Test: {stats.test_size} samples (80/20 split, stratified)
                  </p>

                  {/* Per-class F1 bars */}
                  <div className="space-y-2 mb-5">
                    {stats.classes.map(cls => {
                      const m     = stats.per_class[cls]
                      const color = CLS_COLOR[cls] ?? '#94a3b8'
                      return (
                        <div key={cls}>
                          <div className="flex items-center justify-between mb-1">
                            <div className="flex items-center gap-2">
                              <span style={{ color, fontSize: 12, fontWeight: 600, minWidth: 110 }}>{cls}</span>
                              <span style={{ color: '#475569', fontSize: 11 }}>n={m.support}</span>
                            </div>
                            <div className="flex gap-3" style={{ fontSize: 11 }}>
                              <span style={{ color: '#64748b' }}>P <span style={{ color: '#cbd5e1' }}>{(m.precision * 100).toFixed(0)}%</span></span>
                              <span style={{ color: '#64748b' }}>R <span style={{ color: '#cbd5e1' }}>{(m.recall    * 100).toFixed(0)}%</span></span>
                              <span style={{ color, fontWeight: 700 }}>F1 {(m.f1 * 100).toFixed(0)}%</span>
                            </div>
                          </div>
                          <div className="h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.06)' }}>
                            <div className="h-full rounded-full transition-all duration-700"
                                 style={{ width: `${m.f1 * 100}%`, background: color }} />
                          </div>
                        </div>
                      )
                    })}
                  </div>

                  {/* Confusion matrix */}
                  <div>
                    <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 8 }}>
                      CONFUSION MATRIX — rows=actual, cols=predicted
                    </p>
                    <div style={{ overflowX: 'auto' }}>
                      <table style={{ borderCollapse: 'collapse', fontSize: 11, width: '100%' }}>
                        <thead>
                          <tr>
                            <td style={{ width: 80 }} />
                            {stats.classes.map(cls => (
                              <th key={cls} style={{
                                color: CLS_COLOR[cls] ?? '#94a3b8',
                                fontFamily: 'Orbitron, monospace', fontSize: 8,
                                padding: '2px 4px', textAlign: 'center',
                                maxWidth: 60, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                              }}>
                                {cls.replace('ASSUMED ', 'ASS. ')}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {stats.confusion_matrix.map((row, ri) => {
                            const rowSum  = row.reduce((a, b) => a + b, 0)
                            const actCls  = stats.classes[ri]
                            return (
                              <tr key={ri}>
                                <td style={{
                                  color: CLS_COLOR[actCls] ?? '#94a3b8',
                                  fontFamily: 'Orbitron, monospace', fontSize: 8,
                                  paddingRight: 8, textAlign: 'right', whiteSpace: 'nowrap',
                                }}>
                                  {actCls.replace('ASSUMED ', 'ASS. ')}
                                </td>
                                {row.map((val, ci) => {
                                  const isDiag   = ri === ci
                                  const pct      = rowSum > 0 ? val / rowSum : 0
                                  const bgAlpha  = Math.round(pct * 200)
                                  const bg       = isDiag
                                    ? `${CLS_COLOR[actCls] ?? '#94a3b8'}${bgAlpha.toString(16).padStart(2, '0')}`
                                    : val > 0 ? `rgba(239,68,68,${(val / cmMax * 0.7).toFixed(2)})` : 'transparent'
                                  return (
                                    <td key={ci}
                                        title={`Actual: ${actCls} → Predicted: ${stats.classes[ci]} (${val})`}
                                        style={{
                                          background:  bg,
                                          color:       val > 0 ? (isDiag ? 'white' : '#fca5a5') : '#1e293b',
                                          textAlign:   'center',
                                          padding:     '4px 3px',
                                          borderRadius: 3,
                                          fontWeight:  isDiag ? 700 : 400,
                                          minWidth:    36,
                                        }}>
                                      {val > 0 ? val : '·'}
                                    </td>
                                  )
                                })}
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </>
              )}
            </div>

            {/* Training info */}
            <div className="rounded-xl px-4 py-3"
                 style={{ background: 'rgba(56,189,248,0.04)', border: '1px solid rgba(56,189,248,0.12)' }}>
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 8 }}>TRAINING INFO</p>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1" style={{ fontSize: 12 }}>
                {[
                  ['Framework',      'PyTorch'],
                  ['Data',           'Synthetic (rule-based)'],
                  ['Optimizer',      'Adam'],
                  ['Loss',           'CrossEntropyLoss'],
                  ['Regularization', 'BatchNorm + Dropout 0.2'],
                  ['Input clamp',    '[-10, 10] before forward'],
                ].map(([k, v]) => (
                  <div key={k}>
                    <span style={{ color: '#475569' }}>{k}: </span>
                    <span style={{ color: '#cbd5e1', fontWeight: 500 }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── SENSORS ACTIVE MODAL ─────────────────────────────────────────────── */}
      {sensorsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center"
             style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)' }}
             onClick={() => setSensorsOpen(false)}>
          <div className="relative overflow-y-auto"
               style={{
                 background:   'linear-gradient(135deg, rgba(8,14,26,0.99) 0%, rgba(14,22,40,0.98) 100%)',
                 border:       '1px solid rgba(34,197,94,0.25)', borderTop: '2px solid #22c55e',
                 borderRadius: 14, padding: '24px 28px',
                 maxWidth: 540, width: '90vw', maxHeight: '85vh',
                 boxShadow: '0 24px 80px rgba(0,0,0,0.8)',
                 fontFamily: 'Space Grotesk, sans-serif',
               }}
               onClick={e => e.stopPropagation()}>
            <button onClick={() => setSensorsOpen(false)}
                    style={{ position: 'absolute', top: 14, right: 16, fontSize: 18, background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>✕</button>

            <div className="mb-5">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full" style={{ background: '#22c55e', boxShadow: '0 0 8px #22c55e' }} />
                <span style={{ fontFamily: 'Orbitron, monospace', fontSize: 9, color: '#22c55e', letterSpacing: 3 }}>4 SENSORS ACTIVE</span>
              </div>
              <h2 style={{ fontFamily: 'Orbitron, monospace', fontSize: 15, color: '#e2e8f0', letterSpacing: 2 }}>
                AI Confidence vs Fused Probability
              </h2>
              <p style={{ color: '#64748b', fontSize: 12, marginTop: 3 }}>Why these two numbers are different — and why both matter</p>
            </div>

            {selected && accentStyle && (
              <div className="grid grid-cols-2 gap-3 mb-5">
                <div className="rounded-xl px-4 py-3 text-center"
                     style={{ background: `${accentStyle.color}0d`, border: `1px solid ${accentStyle.color}33` }}>
                  <div style={{ color: '#64748b', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 2, marginBottom: 6 }}>AI CONFIDENCE</div>
                  <div style={{ color: accentStyle.color, fontFamily: 'Orbitron, monospace', fontSize: 26, fontWeight: 900 }}>{aiConf}%</div>
                  <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4 }}>PyTorch → {aiClass}</div>
                </div>
                <div className="rounded-xl px-4 py-3 text-center"
                     style={{ background: 'rgba(56,189,248,0.05)', border: '1px solid rgba(56,189,248,0.2)' }}>
                  <div style={{ color: '#64748b', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 2, marginBottom: 6 }}>FUSED PROBABILITY</div>
                  <div style={{ color: '#38bdf8', fontFamily: 'Orbitron, monospace', fontSize: 26, fontWeight: 900 }}>{fusedProb}%</div>
                  <div style={{ color: '#94a3b8', fontSize: 11, marginTop: 4 }}>Sensors → {fusedBest}</div>
                </div>
                <div className="col-span-2 rounded-lg px-3 py-2 flex items-center justify-center gap-2"
                     style={{ background: 'rgba(245,158,11,0.07)', border: '1px solid rgba(245,158,11,0.2)' }}>
                  <span style={{ color: '#f59e0b', fontSize: 12 }}>Δ Gap:</span>
                  <span style={{ color: '#fbbf24', fontFamily: 'Orbitron, monospace', fontSize: 14, fontWeight: 700 }}>{gap}%</span>
                  <span style={{ color: '#64748b', fontSize: 11 }}>
                    {parseFloat(gap) > 20 ? '— sensors disagree with model' : parseFloat(gap) > 8 ? '— minor divergence' : '— sensors and model agree'}
                  </span>
                </div>
              </div>
            )}

            <div className="space-y-3">
              {[
                { color: '#a78bfa', title: 'AI Confidence — Black box',       body: 'The PyTorch MLP sees all 10 features simultaneously as a single vector and outputs a softmax probability. It learned patterns from training data — it does not know which sensor gave which reading.' },
                { color: '#38bdf8', title: 'Fused Probability — Transparent', body: 'Each sensor (Radar w=0.40, ESM w=0.35, IRST w=0.15, IFF w=0.10) votes independently, then their votes are weighted and summed. Weather degrades IRST confidence in real-time.' },
                { color: '#f59e0b', title: 'Why the gap matters',             body: 'A large gap means the model "learned" a pattern the raw sensors do not fully confirm. In real systems this triggers human review — which is exactly what the Expert Approval button is for.' },
              ].map(({ color, title, body }) => (
                <div key={title} className="rounded-xl px-4 py-3"
                     style={{ background: `${color}08`, border: `1px solid ${color}28`, borderLeft: `3px solid ${color}` }}>
                  <p style={{ color, fontWeight: 700, fontSize: 13, marginBottom: 4 }}>{title}</p>
                  <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{body}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </>
  )
}
