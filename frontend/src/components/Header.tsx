import { useEffect, useState } from 'react'
import type { Track } from '../types'
import { CLASS_STYLES } from '../types'

interface Props {
  pendingCount:  number
  approvedCount: number
  selected:      Track | null
}

const MODEL_CLASSES = [
  { label: 'HOSTILE',        color: '#ef4444', icon: '🚨' },
  { label: 'SUSPECT',        color: '#f59e0b', icon: '⚠️' },
  { label: 'FRIEND',         color: '#22c55e', icon: '🛡️' },
  { label: 'ASSUMED FRIEND', color: '#22c55e', icon: '🤝' },
  { label: 'NEUTRAL',        color: '#94a3b8', icon: '🏳️' },
  { label: 'CIVILIAN',       color: '#38bdf8', icon: '✈️' },
]

const MODEL_FEATURES = [
  { group: 'Kinematic',     items: ['Altitude (ft)', 'Speed (kts)', 'Heading (°)', 'Latitude', 'Longitude'] },
  { group: 'Radar',         items: ['Radar Cross-Section (m²)'] },
  { group: 'Electronic',    items: ['Electronic Signature / IFF'] },
  { group: 'Thermal',       items: ['Thermal Signature (IRST)'] },
  { group: 'Environmental', items: ['Weather condition', 'Flight Profile'] },
]

const GROUP_COLORS: Record<string, string> = {
  Kinematic: '#a78bfa', Radar: '#fb923c', Electronic: '#38bdf8',
  Thermal: '#f472b6', Environmental: '#4ade80',
}

export default function Header({ pendingCount, approvedCount, selected }: Props) {
  const [clock,        setClock]        = useState('')
  const [modelOpen,    setModelOpen]    = useState(false)
  const [sensorsOpen,  setSensorsOpen]  = useState(false)

  useEffect(() => {
    const tick = () => {
      const n   = new Date()
      const pad = (x: number) => String(x).padStart(2, '0')
      setClock(`${pad(n.getUTCHours())}:${pad(n.getUTCMinutes())}:${pad(n.getUTCSeconds())} UTC`)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  // Live values for the sensors modal
  const aiConf     = selected ? (selected.ai_conf * 100).toFixed(1)   : '—'
  const aiClass    = selected?.ai_class ?? '—'
  const fusedProb  = selected ? (((selected.fusion?.probs?.[selected.ai_class]) ?? 0) * 100).toFixed(1) : '—'
  const fusedBest  = selected?.fusion?.best ?? '—'
  const accentStyle = selected ? (CLASS_STYLES[aiClass] ?? CLASS_STYLES['NEUTRAL']) : null
  const gap        = selected ? Math.abs(selected.ai_conf * 100 - ((selected.fusion?.probs?.[selected.ai_class] ?? 0) * 100)).toFixed(1) : '—'

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
          <h1 className="text-lg font-black tracking-widest leading-none"
              style={{ fontFamily: 'Orbitron, monospace', color: '#38bdf8', letterSpacing: 6 }}>
            🛡️ VANGUARD TACTICAL
          </h1>
          <p className="text-xs mt-0.5"
             style={{ color: 'rgba(56,189,248,0.45)', letterSpacing: 3, fontFamily: 'Space Grotesk, sans-serif' }}>
            SENSOR FUSION &amp; THREAT ASSESSMENT SYSTEM
          </p>
        </div>

        <div className="flex items-center gap-6">
          <div className="flex items-center gap-4 text-xs" style={{ fontFamily: 'Space Grotesk, sans-serif' }}>
            {/* Model Online */}
            <button
              onClick={() => setModelOpen(true)}
              className="flex items-center gap-1.5 transition-opacity hover:opacity-80"
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
            >
              <span className="w-2 h-2 rounded-full animate-[pulse-dot_2.2s_ease-in-out_infinite]"
                    style={{ background: '#22c55e', boxShadow: '0 0 6px #22c55e' }} />
              <span style={{ color: '#22c55e', fontWeight: 600 }}>Model Online</span>
            </button>
            {/* Sensors Active */}
            <button
              onClick={() => setSensorsOpen(true)}
              className="flex items-center gap-1.5 transition-opacity hover:opacity-80"
              style={{ background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
            >
              <span className="w-2 h-2 rounded-full animate-[pulse-dot_2.2s_ease-in-out_infinite_0.4s]"
                    style={{ background: '#22c55e', boxShadow: '0 0 6px #22c55e' }} />
              <span style={{ color: '#22c55e', fontWeight: 600 }}>Sensors Active</span>
            </button>
          </div>

          <div className="text-xs text-center" style={{ fontFamily: 'Orbitron, monospace' }}>
            <div className="flex gap-3">
              <span style={{ color: '#f59e0b' }}>{pendingCount} <span style={{ color: '#64748b', fontFamily: 'Space Grotesk' }}>pending</span></span>
              <span style={{ color: '#22c55e' }}>{approvedCount} <span style={{ color: '#64748b', fontFamily: 'Space Grotesk' }}>approved</span></span>
            </div>
          </div>

          <div style={{ fontFamily: 'Orbitron, monospace', fontSize: 11, color: 'rgba(56,189,248,0.5)', letterSpacing: 3 }}>
            {clock}
          </div>
        </div>
      </header>

      {/* ── Model Info Modal ── */}
      {modelOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center"
             style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)' }}
             onClick={() => setModelOpen(false)}>
          <div className="relative overflow-y-auto"
               style={{
                 background: 'linear-gradient(135deg, rgba(8,14,26,0.99) 0%, rgba(14,22,40,0.98) 100%)',
                 border: '1px solid rgba(56,189,248,0.25)', borderTop: '2px solid #38bdf8',
                 borderRadius: 14, padding: '24px 28px',
                 maxWidth: 560, width: '90vw', maxHeight: '85vh',
                 boxShadow: '0 24px 80px rgba(0,0,0,0.8), 0 0 60px rgba(56,189,248,0.08)',
                 fontFamily: 'Space Grotesk, sans-serif',
               }}
               onClick={e => e.stopPropagation()}>
            <button onClick={() => setModelOpen(false)}
                    style={{ position: 'absolute', top: 14, right: 16, fontSize: 18, lineHeight: 1, background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>✕</button>

            <div className="mb-5">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full" style={{ background: '#22c55e', boxShadow: '0 0 8px #22c55e' }} />
                <span style={{ fontFamily: 'Orbitron, monospace', fontSize: 9, color: '#22c55e', letterSpacing: 3 }}>MODEL ONLINE</span>
              </div>
              <h2 style={{ fontFamily: 'Orbitron, monospace', fontSize: 16, color: '#38bdf8', letterSpacing: 3 }}>ImprovedAircraftClassifier</h2>
              <p style={{ color: '#64748b', fontSize: 12, marginTop: 3 }}>PyTorch · 3-layer MLP · 6-class threat classification</p>
            </div>

            <div className="mb-5">
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 10 }}>ARCHITECTURE</p>
              <div className="flex items-center gap-1 flex-wrap">
                {[
                  { label: 'Input',    sub: '10 features',                          color: '#475569' },
                  { label: 'Dense 64', sub: 'BatchNorm · ReLU · Dropout 0.2',       color: '#38bdf8' },
                  { label: 'Dense 32', sub: 'BatchNorm · ReLU · Dropout 0.2',       color: '#38bdf8' },
                  { label: 'Output',   sub: '6 classes · Softmax',                  color: '#22c55e' },
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

            <div className="mb-5">
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 10 }}>INPUT FEATURES (10)</p>
              <div className="space-y-1.5">
                {MODEL_FEATURES.map(({ group, items }) => (
                  <div key={group} className="flex items-start gap-2">
                    <span className="rounded px-1.5 py-0.5 flex-shrink-0"
                          style={{ background: `${GROUP_COLORS[group]}18`, color: GROUP_COLORS[group], fontSize: 10, fontWeight: 600, minWidth: 90, textAlign: 'center' }}>
                      {group}
                    </span>
                    <span style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>{items.join(' · ')}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mb-5">
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 10 }}>OUTPUT CLASSES (6)</p>
              <div className="grid grid-cols-3 gap-2">
                {MODEL_CLASSES.map(({ label, color, icon }) => (
                  <div key={label} className="rounded-lg px-3 py-2 flex items-center gap-2"
                       style={{ background: `${color}10`, border: `1px solid ${color}28` }}>
                    <span style={{ fontSize: 13 }}>{icon}</span>
                    <span style={{ color, fontSize: 11, fontWeight: 600, fontFamily: 'Orbitron, monospace' }}>{label}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl px-4 py-3"
                 style={{ background: 'rgba(56,189,248,0.04)', border: '1px solid rgba(56,189,248,0.12)' }}>
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 8 }}>TRAINING INFO</p>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1" style={{ fontSize: 12 }}>
                {[
                  ['Framework',       'PyTorch'],
                  ['Data',            'Synthetic (rule-based)'],
                  ['Optimizer',       'Adam'],
                  ['Loss',            'CrossEntropyLoss'],
                  ['Regularization',  'BatchNorm + Dropout 0.2'],
                  ['Input clamp',     '[-10, 10] before forward'],
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

      {/* ── Sensors Active Modal ── */}
      {sensorsOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center"
             style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)' }}
             onClick={() => setSensorsOpen(false)}>
          <div className="relative overflow-y-auto"
               style={{
                 background: 'linear-gradient(135deg, rgba(8,14,26,0.99) 0%, rgba(14,22,40,0.98) 100%)',
                 border: '1px solid rgba(34,197,94,0.25)', borderTop: '2px solid #22c55e',
                 borderRadius: 14, padding: '24px 28px',
                 maxWidth: 540, width: '90vw', maxHeight: '85vh',
                 boxShadow: '0 24px 80px rgba(0,0,0,0.8), 0 0 60px rgba(34,197,94,0.06)',
                 fontFamily: 'Space Grotesk, sans-serif',
               }}
               onClick={e => e.stopPropagation()}>
            <button onClick={() => setSensorsOpen(false)}
                    style={{ position: 'absolute', top: 14, right: 16, fontSize: 18, lineHeight: 1, background: 'none', border: 'none', cursor: 'pointer', color: '#64748b' }}>✕</button>

            <div className="mb-5">
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2 h-2 rounded-full" style={{ background: '#22c55e', boxShadow: '0 0 8px #22c55e' }} />
                <span style={{ fontFamily: 'Orbitron, monospace', fontSize: 9, color: '#22c55e', letterSpacing: 3 }}>4 SENSORS ACTIVE</span>
              </div>
              <h2 style={{ fontFamily: 'Orbitron, monospace', fontSize: 15, color: '#e2e8f0', letterSpacing: 2 }}>
                AI Confidence vs Fused Probability
              </h2>
              <p style={{ color: '#64748b', fontSize: 12, marginTop: 3 }}>
                Why these two numbers are different — and why both matter
              </p>
            </div>

            {/* Live values for selected track */}
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

            {/* Explanation */}
            <div className="space-y-3">
              <div className="rounded-xl px-4 py-3"
                   style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.2)', borderLeft: '3px solid #a78bfa' }}>
                <p style={{ color: '#a78bfa', fontWeight: 700, fontSize: 13, marginBottom: 4 }}>AI Confidence — Black box</p>
                <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>
                  The PyTorch MLP sees all 10 features simultaneously as a vector and outputs a
                  softmax probability. It learned patterns from training data — it does not know
                  which sensor gave which reading. The output is a single number per class.
                </p>
              </div>
              <div className="rounded-xl px-4 py-3"
                   style={{ background: 'rgba(56,189,248,0.05)', border: '1px solid rgba(56,189,248,0.2)', borderLeft: '3px solid #38bdf8' }}>
                <p style={{ color: '#38bdf8', fontWeight: 700, fontSize: 13, marginBottom: 4 }}>Fused Probability — Transparent</p>
                <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>
                  Each sensor (Radar w=0.40, ESM w=0.35, IRST w=0.15, IFF w=0.10) votes
                  independently, then their votes are weighted and summed. Weather degrades IRST
                  confidence in real-time. You can see exactly which sensor pulled the number up
                  or down.
                </p>
              </div>
              <div className="rounded-xl px-4 py-3"
                   style={{ background: 'rgba(245,158,11,0.05)', border: '1px solid rgba(245,158,11,0.18)', borderLeft: '3px solid #f59e0b' }}>
                <p style={{ color: '#f59e0b', fontWeight: 700, fontSize: 13, marginBottom: 4 }}>Why the gap matters</p>
                <p style={{ color: '#94a3b8', fontSize: 12, lineHeight: 1.6 }}>
                  A large gap means the model "learned" a pattern the raw sensors don't fully
                  confirm — or vice versa. In real systems this triggers human review. The
                  <span style={{ color: '#fbbf24' }}> Expert Approval</span> button is exactly
                  for this scenario: when AI and sensors diverge, the operator decides.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
