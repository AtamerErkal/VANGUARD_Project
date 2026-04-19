import Plot from 'react-plotly.js'
import type { Track } from '../types'
import { CLASS_STYLES } from '../types'

interface Props { track: Track }

function hdgDelta(a: number, b: number): number {
  const d = Math.abs(a - b)
  return d > 180 ? 360 - d : d
}

export default function Trail3D({ track }: Props) {
  const accent = CLASS_STYLES[track.ai_class]?.color ?? '#38bdf8'
  const n      = track.hist_alts.length

  const alts  = track.hist_alts                               // ft
  const spds  = track.hist_speeds   ?? alts.map(() => track.speed_kts)
  const hdgs  = track.hist_headings ?? alts.map(() => track.heading)
  const times = track.hist_timestamps ?? alts.map((_, i) => `T-${(n - 1 - i) * 5}min`)

  // Heading delta per step (turn rate in degrees)
  const turns = alts.map((_, i) =>
    i === 0 ? 0 : hdgDelta(hdgs[i], hdgs[i - 1])
  )
  const maxTurn = Math.max(...turns, 0.1)

  // Stats
  const altMin = Math.min(...alts), altMax = Math.max(...alts)
  const spdMin = Math.min(...spds), spdMax = Math.max(...spds)
  const peakTurn = Math.max(...turns)

  // Maneuver score: weighted combo of all three dynamics
  const manScore = Math.min(100, Math.round(
    ((altMax - altMin) / 500) +
    ((spdMax - spdMin) / 10)  +
    (peakTurn * 1.5)
  ))

  // Color each point by turn rate: 0=sky, 0.5=amber, 1=red
  const turnNorm = turns.map(t => t / maxTurn)
  const colorscale: [number, string][] = [
    [0,    '#38bdf8'],
    [0.35, '#38bdf8'],
    [0.65, '#f59e0b'],
    [1,    '#ef4444'],
  ]

  // Top-3 maneuver event indices (highest turn rate, excluding first point)
  const eventIdx = [...turns.keys()]
    .filter(i => i > 0)
    .sort((a, b) => turns[b] - turns[a])
    .slice(0, 3)

  // Time index for X axis (0 = oldest)
  const tIdx = alts.map((_, i) => i)

  const altsKft = alts.map(a => +(a / 1000).toFixed(2))

  const statCards = [
    { label: 'ALT Δ',    value: `${((altMax - altMin) / 1000).toFixed(0)} kft`, color: '#38bdf8' },
    { label: 'SPD Δ',    value: `${Math.round(spdMax - spdMin)} kts`,           color: '#a78bfa' },
    { label: 'MAX TURN', value: `${peakTurn.toFixed(0)}°/5min`,                 color: '#fb923c' },
    { label: 'MANEUVER', value: String(manScore),                               color: accent    },
  ]

  return (
    <div className="space-y-3">
      {/* Stats */}
      <div className="grid grid-cols-4 gap-1.5">
        {statCards.map(({ label, value, color }) => (
          <div key={label} className="rounded-lg px-2 py-2 text-center"
               style={{ background: 'rgba(12,18,30,0.9)', border: `1px solid ${color}28` }}>
            <div style={{ color: '#475569', fontSize: 8, fontFamily: 'Orbitron, monospace', letterSpacing: 1.5, marginBottom: 3 }}>
              {label}
            </div>
            <div style={{ color, fontSize: 13, fontWeight: 700, fontFamily: 'Orbitron, monospace', lineHeight: 1 }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      {/* Turn-rate legend */}
      <div className="flex items-center gap-2 px-1">
        <span style={{ color: '#475569', fontSize: 10, fontFamily: 'Orbitron, monospace' }}>TURN RATE</span>
        <div className="flex-1 h-2 rounded-full" style={{
          background: 'linear-gradient(90deg, #38bdf8 0%, #38bdf8 35%, #f59e0b 65%, #ef4444 100%)'
        }} />
        <span style={{ color: '#475569', fontSize: 10 }}>LOW → HIGH</span>
      </div>

      {/* 3D Maneuver Envelope */}
      <Plot
        data={[
          // Trail line
          {
            type: 'scatter3d' as const,
            mode: 'lines' as const,
            x: tIdx,
            y: altsKft,
            z: spds,
            line: {
              color:      turnNorm,
              colorscale,
              width:      5,
              cmin:       0,
              cmax:       1,
            },
            hoverinfo:  'skip' as const,
            showlegend: false,
            name:       'Trail',
          },
          // Nodes
          {
            type: 'scatter3d' as const,
            mode: 'markers' as const,
            x: tIdx,
            y: altsKft,
            z: spds,
            marker: {
              size:       turnNorm.map(t => 4 + t * 6),
              color:      turnNorm,
              colorscale,
              cmin:       0,
              cmax:       1,
              opacity:    0.9,
              line:       { color: 'rgba(0,0,0,0.3)', width: 0.5 },
            },
            customdata: times.map((ts, i) => [ts, altsKft[i], spds[i], hdgs[i], turns[i].toFixed(1)]),
            hovertemplate:
              '<b>%{customdata[0]}</b><br>' +
              'Alt: <b>%{customdata[1]} kft</b>   Speed: <b>%{customdata[2]} kts</b><br>' +
              'Hdg: %{customdata[3]}°   Turn Δ: <b>%{customdata[4]}°</b><extra></extra>',
            showlegend: false,
            name:       'Nodes',
          },
          // Vertical floor projections (altitude "shadow")
          ...tIdx.map(i => ({
            type:   'scatter3d' as const,
            mode:   'lines' as const,
            x:      [i, i],
            y:      [altsKft[i], 0],
            z:      [spds[i], spds[i]],
            line:   { color: `${accent}18`, width: 1 },
            hoverinfo: 'skip' as const,
            showlegend: false,
            name:   '',
          })),
          // Maneuver event markers (top-3 turns)
          {
            type: 'scatter3d' as const,
            mode: 'markers+text' as const,
            x: eventIdx.map(i => tIdx[i]),
            y: eventIdx.map(i => altsKft[i]),
            z: eventIdx.map(i => spds[i]),
            marker: {
              size:   14,
              color:  '#ef4444',
              symbol: 'diamond' as const,
              line:   { color: 'white', width: 1.5 },
              opacity: 1,
            },
            text:         eventIdx.map(i => `${turns[i].toFixed(0)}°`),
            textposition: 'top center' as const,
            textfont:     { color: '#fca5a5', size: 9, family: 'Orbitron' },
            customdata:   eventIdx.map(i => [times[i], altsKft[i], spds[i], turns[i].toFixed(1)]),
            hovertemplate:
              '⚠ MANEUVER EVENT<br>' +
              '<b>%{customdata[0]}</b><br>' +
              'Alt: %{customdata[1]} kft   Speed: %{customdata[2]} kts<br>' +
              'Turn: <b>%{customdata[3]}°</b><extra></extra>',
            showlegend: false,
            name:       'Events',
          },
          // Current position
          {
            type: 'scatter3d' as const,
            mode: 'markers+text' as const,
            x: [tIdx[n - 1]],
            y: [altsKft[n - 1]],
            z: [spds[n - 1]],
            marker: {
              size:    12,
              color:   accent,
              symbol:  'circle' as const,
              line:    { color: 'white', width: 2 },
              opacity: 1,
            },
            text:         ['NOW'],
            textposition: 'top center' as const,
            textfont:     { color: 'white', size: 9, family: 'Orbitron' },
            hoverinfo:    'skip' as const,
            showlegend:   false,
            name:         'Current',
          },
        ]}
        layout={{
          height:        340,
          paper_bgcolor: 'rgba(0,0,0,0)',
          scene: {
            bgcolor: 'rgba(6,10,16,0.95)',
            xaxis: {
              title:          { text: 'Time →', font: { color: '#475569', size: 10 } },
              tickvals:       tIdx,
              ticktext:       times.map((t, i) => i % 3 === 0 ? t : ''),
              tickfont:       { color: '#334155', size: 8 },
              color:          '#1e293b',
              gridcolor:      '#1e293b',
              showbackground: false,
              zeroline:       false,
            },
            yaxis: {
              title:          { text: 'Alt (kft)', font: { color: '#475569', size: 10 } },
              tickfont:       { color: '#334155', size: 8 },
              color:          '#1e293b',
              gridcolor:      '#1e293b',
              showbackground: false,
              zeroline:       false,
            },
            zaxis: {
              title:          { text: 'Speed (kts)', font: { color: '#475569', size: 10 } },
              tickfont:       { color: '#334155', size: 8 },
              color:          '#1e293b',
              gridcolor:      '#1e293b',
              showbackground: false,
              zeroline:       false,
            },
            camera: { eye: { x: 1.8, y: -1.6, z: 1.1 } },
            aspectmode: 'manual' as const,
            aspectratio: { x: 1.4, y: 1, z: 0.9 },
          },
          margin:     { t: 4, b: 4, l: 0, r: 0 },
          showlegend: false,
          annotations: [{
            text:      `${track.track_id} · ${n} waypoints · ${(n - 1) * 5} min history`,
            showarrow: false,
            x: 0, y: 0,
            xref: 'paper' as const, yref: 'paper' as const,
            xanchor: 'left' as const, yanchor: 'bottom' as const,
            font: { color: '#1e293b', size: 9, family: 'Space Grotesk' },
          }],
        }}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: '100%' }}
      />

      {/* Maneuver event list */}
      {eventIdx.length > 0 && (
        <div className="space-y-1">
          <p style={{ color: '#475569', fontSize: 9, fontFamily: 'Orbitron, monospace', letterSpacing: 2 }}>
            TOP MANEUVER EVENTS
          </p>
          {eventIdx.map((i, rank) => (
            <div key={i} className="flex items-center gap-3 px-3 py-1.5 rounded-lg"
                 style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)' }}>
              <span style={{ color: '#ef4444', fontFamily: 'Orbitron, monospace', fontSize: 11, minWidth: 16 }}>
                #{rank + 1}
              </span>
              <span style={{ color: '#94a3b8', fontSize: 11, flex: 1 }}>{times[i]}</span>
              <span style={{ color: '#fca5a5', fontSize: 11 }}>
                {altsKft[i]} kft · {spds[i]} kts
              </span>
              <span style={{ color: '#ef4444', fontFamily: 'Orbitron, monospace', fontSize: 11, minWidth: 40, textAlign: 'right' }}>
                {turns[i].toFixed(0)}° turn
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
