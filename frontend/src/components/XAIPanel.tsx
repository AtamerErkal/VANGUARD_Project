import type { XAIItem } from '../types'
import { CLASS_STYLES } from '../types'

interface Props {
  xai:      XAIItem[]
  aiClass:  string
}

const GROUP_COLORS: Record<string, string> = {
  Electronic:    '#38bdf8',
  Kinematic:     '#a78bfa',
  Radar:         '#fb923c',
  Thermal:       '#f472b6',
  Environmental: '#4ade80',
}

const DIR_ICON: Record<string, string> = {
  supporting:  '↑',
  conflicting: '↓',
  neutral:     '→',
}

export default function XAIPanel({ xai, aiClass }: Props) {
  const style       = CLASS_STYLES[aiClass] ?? CLASS_STYLES['NEUTRAL']
  const supporting  = xai.filter(x => x.direction === 'supporting')
  const conflicting = xai.filter(x => x.direction === 'conflicting')
  const maxImp      = Math.max(...xai.map(x => x.importance), 0.01)

  return (
    <div className="space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <p style={{ color: '#38bdf8aa', fontFamily: 'Orbitron, monospace', fontSize: 11, letterSpacing: 3, textTransform: 'uppercase', fontWeight: 600 }}>
          Explainable AI
        </p>
        <span className="px-2 py-1 rounded font-bold"
              style={{ background: style.bg, color: style.color, fontFamily: 'Orbitron, monospace', fontSize: 11, letterSpacing: 1 }}>
          WHY {aiClass}?
        </span>
      </div>

      {/* Summary row */}
      <div className="flex gap-2">
        <div className="flex-1 rounded-lg px-3 py-2 text-center"
             style={{ background: 'rgba(34,197,94,0.1)', border: '1px solid rgba(34,197,94,0.3)' }}>
          <span style={{ color: '#4ade80', fontWeight: 700, fontSize: 15 }}>{supporting.length}</span>
          <span style={{ color: '#86efac', fontSize: 13, marginLeft: 5 }}>supporting</span>
        </div>
        <div className="flex-1 rounded-lg px-3 py-2 text-center"
             style={{ background: 'rgba(239,68,68,0.09)', border: '1px solid rgba(239,68,68,0.3)' }}>
          <span style={{ color: '#f87171', fontWeight: 700, fontSize: 15 }}>{conflicting.length}</span>
          <span style={{ color: '#fca5a5', fontSize: 13, marginLeft: 5 }}>conflicting</span>
        </div>
      </div>

      {/* Feature importance bars */}
      <div className="space-y-2.5">
        {xai.map((item) => {
          const barColor = item.direction === 'supporting'
            ? GROUP_COLORS[item.group] ?? style.color
            : item.direction === 'conflicting' ? '#ef4444' : '#334155'
          const barWidth = (item.importance / maxImp) * 100
          const opacity  = item.direction === 'neutral' ? 0.45 : 1

          return (
            <div key={item.feature} style={{ opacity }}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2 min-w-0">
                  <span style={{ color: barColor, fontSize: 14, width: 14, flexShrink: 0 }}>
                    {DIR_ICON[item.direction]}
                  </span>
                  <span className="truncate" style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 500 }}>
                    {item.label}
                  </span>
                  <span className="px-1.5 py-0.5 rounded flex-shrink-0"
                        style={{ background: 'rgba(255,255,255,0.07)', color: GROUP_COLORS[item.group] ?? '#94a3b8', fontSize: 11, fontWeight: 600 }}>
                    {item.group}
                  </span>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0 ml-2">
                  <span style={{ color: '#cbd5e1', fontSize: 12 }}>{item.value}</span>
                  <span style={{ color: barColor, fontSize: 13, fontWeight: 700, minWidth: 34, textAlign: 'right' }}>
                    {(item.importance * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="h-2 rounded-full" style={{ background: 'rgba(255,255,255,0.06)' }}>
                <div className="h-full rounded-full transition-all duration-700"
                     style={{ width: `${barWidth}%`, background: barColor }} />
              </div>
            </div>
          )
        })}
      </div>

      {/* Human-readable explanation */}
      <div className="rounded-xl px-3 py-3 space-y-1.5"
           style={{ background: 'rgba(12,18,30,0.8)', border: '1px solid rgba(56,189,248,0.12)' }}>
        <p style={{ color: '#cbd5e1', fontStyle: 'italic', fontSize: 13 }}>
          Top factors driving <span style={{ color: style.color, fontWeight: 700 }}>{aiClass}</span>:
        </p>
        {supporting.slice(0, 3).map(item => (
          <p key={item.feature} style={{ color: '#94a3b8', fontSize: 12 }}>
            <span style={{ color: '#4ade80', marginRight: 4 }}>↑</span>
            <span style={{ color: '#e2e8f0', fontWeight: 500 }}>{item.label}</span>
            <span style={{ color: '#94a3b8' }}> ({item.value}) — {(item.importance * 100).toFixed(0)}% weight</span>
          </p>
        ))}
        {conflicting.length > 0 && (
          <p style={{ color: '#94a3b8', fontSize: 12, marginTop: 6 }}>
            <span style={{ color: '#f87171', fontWeight: 600 }}>↓ Conflicting: </span>
            <span style={{ color: '#fca5a5' }}>{conflicting.map(x => x.label).join(', ')}</span>
          </p>
        )}
      </div>
    </div>
  )
}
