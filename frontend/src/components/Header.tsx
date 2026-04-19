import { useEffect, useState } from 'react'

interface Props {
  pendingCount:  number
  approvedCount: number
}

export default function Header({ pendingCount, approvedCount }: Props) {
  const [clock, setClock] = useState('')

  useEffect(() => {
    const tick = () => {
      const n = new Date()
      const pad = (x: number) => String(x).padStart(2, '0')
      setClock(`${pad(n.getUTCHours())}:${pad(n.getUTCMinutes())}:${pad(n.getUTCSeconds())} UTC`)
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  return (
    <header
      className="relative flex items-center justify-between px-6 py-3 overflow-hidden"
      style={{
        background: 'linear-gradient(135deg, rgba(15,23,42,0.97) 0%, rgba(30,41,59,0.9) 100%)',
        borderBottom: '1px solid rgba(56,189,248,0.18)',
        boxShadow: '0 0 40px rgba(56,189,248,0.06)',
      }}
    >
      {/* scan line */}
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, #38bdf8, transparent)', animation: 'scan 3s ease-in-out infinite' }}
      />

      <div>
        <h1
          className="text-lg font-black tracking-widest leading-none"
          style={{ fontFamily: 'Orbitron, monospace', color: '#38bdf8', letterSpacing: 6 }}
        >
          🛡️ VANGUARD TACTICAL
        </h1>
        <p className="text-xs mt-0.5" style={{ color: 'rgba(56,189,248,0.45)', letterSpacing: 3, fontFamily: 'Space Grotesk, sans-serif' }}>
          SENSOR FUSION &amp; THREAT ASSESSMENT SYSTEM
        </p>
      </div>

      <div className="flex items-center gap-6">
        {/* Status dots */}
        <div className="flex items-center gap-4 text-xs" style={{ fontFamily: 'Space Grotesk, sans-serif', color: '#475569' }}>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full animate-[pulse-dot_2.2s_ease-in-out_infinite]" style={{ background: '#22c55e', boxShadow: '0 0 6px #22c55e' }} />
            Model Online
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full animate-[pulse-dot_2.2s_ease-in-out_infinite_0.4s]" style={{ background: '#22c55e', boxShadow: '0 0 6px #22c55e' }} />
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
  )
}
