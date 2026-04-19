import { useState } from 'react'
import type { Track, ApprovalState } from '../types'
import { CLASS_STYLES } from '../types'

interface Props {
  track:    Track
  approval: ApprovalState | null
  onDecide: (id: string, state: ApprovalState) => void
}

// NATO standard order: threat descending
const CLASSES = ['HOSTILE', 'SUSPECT', 'UNKNOWN', 'NEUTRAL', 'ASSUMED FRIEND', 'FRIEND']

export default function ExpertApproval({ track, approval, onDecide }: Props) {
  const [overrideClass, setOverrideClass] = useState(track.ai_class)

  return (
    <div className="rounded-xl px-3 py-3 space-y-3" style={{ background: 'rgba(12,18,30,0.7)', border: '1px solid rgba(56,189,248,0.1)' }}>
      <p className="text-xs font-semibold tracking-widest uppercase" style={{ color: '#38bdf888', fontFamily: 'Orbitron, monospace' }}>
        Expert Decision
      </p>

      {approval && (
        <div className="text-xs rounded-lg px-2 py-1.5" style={{
          background: approval.action === 'approved' ? 'rgba(34,197,94,0.08)' : 'rgba(245,158,11,0.08)',
          border:     `1px solid ${approval.action === 'approved' ? 'rgba(34,197,94,0.3)' : 'rgba(245,158,11,0.3)'}`,
          color:      approval.action === 'approved' ? '#4ade80' : '#fbbf24',
        }}>
          {approval.action === 'approved'
            ? '✓ Classification approved'
            : `↺ Overridden → ${approval.override_class}`}
        </div>
      )}

      <div className="flex gap-2">
        <button
          onClick={() => onDecide(track.track_id, { action: 'approved' })}
          className="flex-1 text-xs font-bold py-2 rounded-lg transition-all duration-150 hover:brightness-110"
          style={{ background: 'rgba(34,197,94,0.15)', border: '1px solid rgba(34,197,94,0.4)', color: '#4ade80', fontFamily: 'Orbitron, monospace', letterSpacing: 1 }}
        >
          ✅ APPROVE
        </button>

        <select
          value={overrideClass}
          onChange={e => setOverrideClass(e.target.value)}
          className="flex-1 text-xs py-2 px-2 rounded-lg outline-none"
          style={{ background: 'rgba(12,18,30,0.9)', border: '1px solid rgba(56,189,248,0.2)', color: '#94a3b8', fontFamily: 'Space Grotesk, sans-serif' }}
        >
          {CLASSES.map(c => (
            <option key={c} value={c} style={{ background: '#0d1117' }}>{c}</option>
          ))}
        </select>

        <button
          onClick={() => onDecide(track.track_id, { action: 'override', override_class: overrideClass })}
          className="flex-1 text-xs font-bold py-2 rounded-lg transition-all duration-150 hover:brightness-110"
          style={{ background: 'rgba(245,158,11,0.12)', border: '1px solid rgba(245,158,11,0.35)', color: '#fbbf24', fontFamily: 'Orbitron, monospace', letterSpacing: 1 }}
        >
          ↺ OVERRIDE
        </button>
      </div>
    </div>
  )
}
