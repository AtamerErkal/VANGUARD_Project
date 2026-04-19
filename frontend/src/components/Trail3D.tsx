import Plot from 'react-plotly.js'
import type { Track } from '../types'
import { CLASS_STYLES } from '../types'

interface Props { track: Track }

export default function Trail3D({ track }: Props) {
  const accentColor = CLASS_STYLES[track.ai_class]?.color ?? '#38bdf8'
  const n           = track.hist_lats.length
  const altsKft     = track.hist_alts.map(a => +(a / 1000).toFixed(2))
  const timestamps  = track.hist_timestamps ?? track.hist_lats.map((_, i) => `T-${(n - 1 - i) * 5}min`)

  // Normalize time index 0→1 for colorscale (oldest=0, current=1)
  const timeNorm    = track.hist_lats.map((_, i) => i / (n - 1))

  // History points (all but last)
  const histN = n - 1

  return (
    <Plot
      data={[
        // Trail line
        {
          type:   'scatter3d' as const,
          mode:   'lines' as const,
          x:      track.hist_lons.slice(0, histN),
          y:      track.hist_lats.slice(0, histN),
          z:      altsKft.slice(0, histN),
          line:   {
            color:     timeNorm.slice(0, histN),
            colorscale: [
              [0,   'rgba(56,189,248,0.15)'],
              [0.5, 'rgba(56,189,248,0.55)'],
              [1,   '#38bdf8'],
            ],
            width: 4,
          },
          name:          'Trail',
          hoverinfo:     'skip' as const,
          showlegend:    false,
        },
        // History nodes (colored by time)
        {
          type:   'scatter3d' as const,
          mode:   'markers' as const,
          x:      track.hist_lons.slice(0, histN),
          y:      track.hist_lats.slice(0, histN),
          z:      altsKft.slice(0, histN),
          marker: {
            size:       timeNorm.slice(0, histN).map(t => 3 + t * 3),
            color:      timeNorm.slice(0, histN),
            colorscale: [
              [0,   'rgba(56,189,248,0.2)'],
              [1,   '#38bdf8'],
            ],
            opacity: 0.85,
          },
          customdata: timestamps.slice(0, histN).map((ts, i) => [ts, altsKft[i]]),
          hovertemplate:
            '<b>%{customdata[0]}</b><br>' +
            'Lon: %{x:.3f}  Lat: %{y:.3f}<br>' +
            'Alt: %{customdata[1]} kft<extra></extra>',
          name:       'History',
          showlegend: false,
        },
        // Current position
        {
          type:   'scatter3d' as const,
          mode:   'markers+text' as const,
          x:      [track.hist_lons[n - 1]],
          y:      [track.hist_lats[n - 1]],
          z:      [altsKft[n - 1]],
          marker: {
            size:   12,
            color:  accentColor,
            opacity: 1,
            symbol: 'circle' as const,
            line:   { color: 'white', width: 1.5 },
          },
          text:         [track.track_id],
          textposition: 'top center' as const,
          textfont:     { color: 'white', size: 10, family: 'Orbitron, monospace' },
          customdata:   [[timestamps[n - 1], altsKft[n - 1]]],
          hovertemplate:
            `<b>${track.track_id}</b><br>` +
            '<b>NOW</b> — %{customdata[0]}<br>' +
            'Alt: %{customdata[1]} kft<extra></extra>',
          name:       'Current',
          showlegend: false,
        },
      ]}
      layout={{
        height:        320,
        paper_bgcolor: 'rgba(0,0,0,0)',
        scene: {
          bgcolor: 'rgba(6,10,16,0.95)',
          xaxis: {
            title: { text: 'Longitude', font: { color: '#475569', size: 10 } },
            color: '#334155', gridcolor: '#1e293b', showbackground: false, zeroline: false,
          },
          yaxis: {
            title: { text: 'Latitude', font: { color: '#475569', size: 10 } },
            color: '#334155', gridcolor: '#1e293b', showbackground: false, zeroline: false,
          },
          zaxis: {
            title: { text: 'Alt (kft)', font: { color: '#475569', size: 10 } },
            color: '#334155', gridcolor: '#1e293b', showbackground: false, zeroline: false,
          },
          camera: { eye: { x: 1.5, y: -1.5, z: 1.2 } },
        },
        margin:    { t: 0, b: 0, l: 0, r: 0 },
        showlegend: false,
        annotations: [{
          text:    `Track ${track.track_id} · ${track.hist_lats.length} waypoints · ${(track.hist_lats.length - 1) * 5} min history`,
          showarrow: false,
          x: 0, y: 0,
          xref: 'paper' as const, yref: 'paper' as const,
          xanchor: 'left' as const, yanchor: 'bottom' as const,
          font: { color: '#334155', size: 9, family: 'Space Grotesk' },
        }],
      }}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%' }}
    />
  )
}
