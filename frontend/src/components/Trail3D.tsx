import Plot from 'react-plotly.js'
import type { Track } from '../types'
import { CLASS_STYLES } from '../types'

interface Props { track: Track }

export default function Trail3D({ track }: Props) {
  const color   = CLASS_STYLES[track.ai_class]?.color ?? '#38bdf8'
  const histAlts = track.hist_alts.map(a => a / 1000)

  return (
    <Plot
      data={[
        {
          type:   'scatter3d' as const,
          mode:   'lines+markers' as const,
          x:      track.hist_lons.slice(0, -1),
          y:      track.hist_lats.slice(0, -1),
          z:      histAlts.slice(0, -1),
          line:   { color: '#38bdf8', width: 3 },
          marker: { size: 3, color: '#38bdf8', opacity: 0.6 },
          name:   'Trail',
          hovertemplate: 'Lon: %{x:.3f}<br>Lat: %{y:.3f}<br>Alt: %{z:.1f}k ft<extra></extra>',
        },
        {
          type:   'scatter3d' as const,
          mode:   'markers+text' as const,
          x:      [track.hist_lons[track.hist_lons.length - 1]],
          y:      [track.hist_lats[track.hist_lats.length - 1]],
          z:      [histAlts[histAlts.length - 1]],
          marker: { size: 9, color, opacity: 1, line: { color: 'white', width: 1 } },
          text:   [track.track_id],
          textposition: 'top center' as const,
          textfont: { color: 'white', size: 10 },
          name:   'Current',
          hovertemplate: `${track.track_id}<br>Alt: %{z:.1f}k ft<extra></extra>`,
        },
      ]}
      layout={{
        height:          300,
        paper_bgcolor:   'rgba(0,0,0,0)',
        scene: {
          bgcolor: 'rgba(6,10,16,0.95)',
          xaxis:   { title: 'Lon', color: '#475569', gridcolor: '#1e293b', showbackground: false },
          yaxis:   { title: 'Lat', color: '#475569', gridcolor: '#1e293b', showbackground: false },
          zaxis:   { title: 'Alt (kft)', color: '#475569', gridcolor: '#1e293b', showbackground: false },
        },
        margin:          { t: 0, b: 0, l: 0, r: 0 },
        showlegend:      false,
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
