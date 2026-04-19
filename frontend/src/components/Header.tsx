import { useEffect, useState } from 'react'

interface Props {
  pendingCount:  number
  approvedCount: number
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
  { group: 'Kinematic',    items: ['Altitude (ft)', 'Speed (kts)', 'Heading (°)', 'Latitude', 'Longitude'] },
  { group: 'Radar',        items: ['Radar Cross-Section (m²)'] },
  { group: 'Electronic',   items: ['Electronic Signature / IFF'] },
  { group: 'Thermal',      items: ['Thermal Signature (IRST)'] },
  { group: 'Environmental',items: ['Weather condition', 'Flight Profile'] },
]

const GROUP_COLORS: Record<string, string> = {
  Kinematic: '#a78bfa', Radar: '#fb923c', Electronic: '#38bdf8',
  Thermal: '#f472b6', Environmental: '#4ade80',
}

export default function Header({ pendingCount, approvedCount }: Props) {
  const [clock,      setClock]      = useState('')
  const [modelOpen,  setModelOpen]  = useState(false)

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

  return (
    <>
      <header
        className="relative flex items-center justify-between px-6 py-3 overflow-hidden"
        style={{
          background:   'linear-gradient(135deg, rgba(15,23,42,0.97) 0%, rgba(30,41,59,0.9) 100%)',
          borderBottom: '1px solid rgba(56,189,248,0.18)',
          boxShadow:    '0 0 40px rgba(56,189,248,0.06)',
        }}
      >
        {/* scan line */}
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
          {/* Status dots */}
          <div className="flex items-center gap-4 text-xs" style={{ fontFamily: 'Space Grotesk, sans-serif', color: '#475569' }}>
            {/* Clickable Model Online */}
            <button
              onClick={() => setModelOpen(true)}
              className="flex items-center gap-1.5 transition-all hover:opacity-80 rounded-lg px-2 py-1 -mx-2 -my-1"
              style={{ background: 'transparent', border: 'none', cursor: 'pointer' }}
              title="View model details"
            >
              <span className="w-2 h-2 rounded-full animate-[pulse-dot_2.2s_ease-in-out_infinite]"
                    style={{ background: '#22c55e', boxShadow: '0 0 6px #22c55e' }} />
              <span style={{ color: '#22c55e', fontWeight: 600 }}>Model Online</span>
            </button>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full animate-[pulse-dot_2.2s_ease-in-out_infinite_0.4s]"
                    style={{ background: '#22c55e', boxShadow: '0 0 6px #22c55e' }} />
              Sensors Active
            </div>
          </div>

          {/* Queue badge */}
          <div className="text-xs text-center" style={{ fontFamily: 'Orbitron, monospace' }}>
            <div className="flex gap-3">
              <span style={{ color: '#f59e0b' }}>{pendingCount} <span style={{ color: '#475569', fontFamily: 'Space Grotesk' }}>pending</span></span>
              <span style={{ color: '#22c55e' }}>{approvedCount} <span style={{ color: '#475569', fontFamily: 'Space Grotesk' }}>approved</span></span>
            </div>
          </div>

          {/* Clock */}
          <div style={{ fontFamily: 'Orbitron, monospace', fontSize: 11, color: 'rgba(56,189,248,0.5)', letterSpacing: 3 }}>
            {clock}
          </div>
        </div>
      </header>

      {/* Model Info Modal */}
      {modelOpen && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center"
          style={{ background: 'rgba(0,0,0,0.75)', backdropFilter: 'blur(4px)' }}
          onClick={() => setModelOpen(false)}
        >
          <div
            className="relative overflow-y-auto"
            style={{
              background:     'linear-gradient(135deg, rgba(8,14,26,0.99) 0%, rgba(14,22,40,0.98) 100%)',
              border:         '1px solid rgba(56,189,248,0.25)',
              borderTop:      '2px solid #38bdf8',
              borderRadius:   14,
              padding:        '24px 28px',
              maxWidth:       560,
              width:          '90vw',
              maxHeight:      '85vh',
              boxShadow:      '0 24px 80px rgba(0,0,0,0.8), 0 0 60px rgba(56,189,248,0.08)',
              fontFamily:     'Space Grotesk, sans-serif',
            }}
            onClick={e => e.stopPropagation()}
          >
            {/* Close */}
            <button onClick={() => setModelOpen(false)}
                    className="absolute top-4 right-4 text-slate-500 hover:text-slate-300 transition-colors"
                    style={{ fontSize: 18, lineHeight: 1, background: 'none', border: 'none', cursor: 'pointer' }}>
              ✕
            </button>

            {/* Header */}
            <div className="mb-5">
              <div className="flex items-center gap-3 mb-1">
                <span className="w-2 h-2 rounded-full" style={{ background: '#22c55e', boxShadow: '0 0 8px #22c55e', flexShrink: 0 }} />
                <span style={{ fontFamily: 'Orbitron, monospace', fontSize: 9, color: '#22c55e', letterSpacing: 3 }}>MODEL ONLINE</span>
              </div>
              <h2 style={{ fontFamily: 'Orbitron, monospace', fontSize: 16, color: '#38bdf8', letterSpacing: 3 }}>
                ImprovedAircraftClassifier
              </h2>
              <p style={{ color: '#64748b', fontSize: 12, marginTop: 3 }}>
                PyTorch · 3-layer MLP · 6-class threat classification
              </p>
            </div>

            {/* Architecture */}
            <div className="mb-5">
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 10 }}>
                ARCHITECTURE
              </p>
              <div className="flex items-center gap-1 flex-wrap">
                {[
                  { label: 'Input', sub: '10 features', color: '#475569' },
                  { label: 'Dense 64', sub: 'BatchNorm · ReLU · Dropout 0.2', color: '#38bdf8' },
                  { label: 'Dense 32', sub: 'BatchNorm · ReLU · Dropout 0.2', color: '#38bdf8' },
                  { label: 'Output', sub: '6 classes · Softmax', color: '#22c55e' },
                ].map((layer, i, arr) => (
                  <div key={layer.label} className="flex items-center gap-1">
                    <div className="rounded-lg px-3 py-2 text-center"
                         style={{ background: 'rgba(255,255,255,0.04)', border: `1px solid ${layer.color}33`, minWidth: 90 }}>
                      <div style={{ color: layer.color, fontFamily: 'Orbitron, monospace', fontSize: 11, fontWeight: 700 }}>{layer.label}</div>
                      <div style={{ color: '#475569', fontSize: 10, marginTop: 2 }}>{layer.sub}</div>
                    </div>
                    {i < arr.length - 1 && <span style={{ color: '#334155', fontSize: 16 }}>→</span>}
                  </div>
                ))}
              </div>
            </div>

            {/* Features */}
            <div className="mb-5">
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 10 }}>
                INPUT FEATURES (10)
              </p>
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

            {/* Classes */}
            <div className="mb-5">
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 10 }}>
                OUTPUT CLASSES (6)
              </p>
              <div className="grid grid-cols-3 gap-2">
                {MODEL_CLASSES.map(({ label, color, icon }) => (
                  <div key={label} className="rounded-lg px-3 py-2 flex items-center gap-2"
                       style={{ background: `${color}10`, border: `1px solid ${color}28` }}>
                    <span style={{ fontSize: 13 }}>{icon}</span>
                    <span style={{ color, fontSize: 11, fontWeight: 600, fontFamily: 'Orbitron, monospace', letterSpacing: 0.5 }}>{label}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Training info */}
            <div className="rounded-xl px-4 py-3"
                 style={{ background: 'rgba(56,189,248,0.04)', border: '1px solid rgba(56,189,248,0.12)' }}>
              <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 3, marginBottom: 8 }}>
                TRAINING INFO
              </p>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1" style={{ fontSize: 12 }}>
                {[
                  ['Framework',    'PyTorch'],
                  ['Data',         'Synthetic (rule-based)'],
                  ['Optimizer',    'Adam'],
                  ['Loss',         'CrossEntropyLoss'],
                  ['Regularization','BatchNorm + Dropout 0.2'],
                  ['Input clamp',  '[-10, 10] before forward'],
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
    </>
  )
}
