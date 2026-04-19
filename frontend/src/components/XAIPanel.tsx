import type { XAIItem } from '../types'
import { CLASS_STYLES } from '../types'

interface Props {
  xai:      XAIItem[]
  aiClass:  string
}

const GROUP_COLORS: Record<string, string> = {
  Electronic:   '#38bdf8',
  Kinematic:    '#a78bfa',
  Radar:        '#fb923c',
  Thermal:      '#f472b6',
  Environmental:'#4ade80',
}

const DIR_ICON: Record<string, string> = {
  supporting:  '↑',
  conflicting: '↓',
  neutral:     '→',
}

export default function XAIPanel({ xai, aiClass }: Props) {
  const style      = CLASS_STYLES[aiClass] ?? CLASS_STYLES['NEUTRAL']
  const supporting = xai.filter(x => x.direction === 'supporting')
  const conflicting = xai.filter(x => x.direction === 'conflicting')
  const maxImp     = Math.max(...xai.map(x => x.importance), 0.01)

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold tracking-widest uppercase"
           style={{ color: '#38bdf888', fontFamily: 'Orbitron, monospace' }}>
          Explainable AI
        </p>
        <span className="text-xs px-2 py-0.5 rounded"
              style={{ background: style.bg, color: style.color, fontFamily: 'Orbitron, monospace', fontSize: 9, letterSpacing: 1 }}>
          WHY {aiClass}?
        </span>
      </div>

      {/* Summary row */}
      <div className="flex gap-2 text-xs">
        <div className="flex-1 rounded-lg px-2 py-1.5 text-center"
             style={{ background: 'rgba(34,197,94,0.08)', border: '1px solid rgba(34,197,94,0.2)' }}>
          <span style={{ color: '#4ade80', fontWeight: 700 }}>{supporting.length}</span>
          <span style={{ color: '#475569' }}> supporting</span>
        </div>
        <div className="flex-1 rounded-lg px-2 py-1.5 text-center"
             style={{ background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.2)' }}>
          <span style={{ color: '#f87171', fontWeight: 700 }}>{conflicting.length}</span>
          <span style={{ color: '#475569' }}> conflicting</span>
        </div>
      </div>

      {/* Feature importance bars */}
      <div className="space-y-2">
        {xai.map((item) => {
          const barColor  = item.direction === 'supporting'
            ? GROUP_COLORS[item.group] ?? style.color
            : item.direction === 'conflicting' ? '#ef4444' : '#334155'
          const barWidth  = (item.importance / maxImp) * 100
          const opacity   = item.direction === 'neutral' ? 0.4 : 1

          return (
            <div key={item.feature} style={{ opacity }}>
              <div className="flex items-center justify-between mb-0.5">
                <div className="flex items-center gap-1.5">
                  <span className="text-xs w-2" style={{ color: barColor }}>{DIR_ICON[item.direction]}</span>
                  <span className="text-xs truncate max-w-[160px]" style={{ color: '#94a3b8' }}>
                    {item.label}
                  </span>
                  <span className="text-xs px-1 rounded" style={{ background: 'rgba(255,255,255,0.04)', color: '#475569', fontSize: 9 }}>
                    {item.group}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-xs" style={{ color: '#475569', fontSize: 9 }}>{item.value}</span>
                  <span className="text-xs font-bold w-8 text-right" style={{ color: barColor }}>
                    {(item.importance * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.05)' }}>
                <div
                  className="h-full rounded-full transition-all duration-700"
                  style={{ width: `${barWidth}%`, background: barColor }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {/* Human-readable explanation */}
      <div className="rounded-xl px-3 py-2.5 text-xs space-y-1"
           style={{ background: 'rgba(12,18,30,0.7)', border: '1px solid rgba(56,189,248,0.08)' }}>
        <p style={{ color: '#64748b', fontStyle: 'italic' }}>
          Top factors driving <span style={{ color: style.color }}>{aiClass}</span>:
        </p>
        {supporting.slice(0, 3).map(item => (
          <p key={item.feature} style={{ color: '#475569' }}>
            <span style={{ color: '#4ade80' }}>↑</span>{' '}
            <span style={{ color: '#94a3b8' }}>{item.label}</span>
            {' '}({item.value}) — {(item.importance * 100).toFixed(0)}% weight
          </p>
        ))}
        {conflicting.length > 0 && (
          <p style={{ color: '#475569', marginTop: 4 }}>
            <span style={{ color: '#f87171' }}>↓ Conflicting: </span>
            {conflicting.map(x => x.label).join(', ')}
          </p>
        )}
      </div>
    </div>
  )
}
