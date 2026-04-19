import { useRef, useCallback } from 'react'
import Map, { Marker, Popup } from 'react-map-gl/maplibre'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { Track } from '../types'
import { CLASS_STYLES } from '../types'

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'

interface Props {
  tracks:     Track[]
  selectedId: string | null
  onSelect:   (id: string) => void
}

function AircraftMarker({ track, selected, onSelect }: {
  track:    Track
  selected: boolean
  onSelect: (id: string) => void
}) {
  const style  = CLASS_STYLES[track.ai_class] ?? CLASS_STYLES['NEUTRAL']
  const isHostile = ['HOSTILE', 'SUSPECT'].includes(track.ai_class)

  return (
    <Marker
      latitude={track.latitude}
      longitude={track.longitude}
      anchor="center"
      onClick={e => { e.originalEvent.stopPropagation(); onSelect(track.track_id) }}
    >
      <div
        className="cursor-pointer transition-transform duration-150"
        style={{ transform: selected ? 'scale(1.35)' : 'scale(1)' }}
        title={`${track.track_id} — ${track.ai_class} (${(track.ai_conf * 100).toFixed(0)}%)`}
      >
        {/* Pulse ring for hostile/suspect */}
        {isHostile && (
          <span
            className="absolute inset-0 rounded-full animate-ping opacity-40"
            style={{ background: style.color }}
          />
        )}
        <div
          className="relative flex items-center justify-center rounded-full border-2 text-xs font-bold"
          style={{
            width:       selected ? 38 : 30,
            height:      selected ? 38 : 30,
            background:  style.bg,
            borderColor: style.color,
            color:       style.color,
            boxShadow:   selected ? `0 0 16px ${style.color}88` : `0 0 6px ${style.color}44`,
            fontSize:    12,
          }}
        >
          {style.icon}
        </div>
        {/* Track ID label */}
        <div
          className="absolute top-full left-1/2 -translate-x-1/2 mt-1 whitespace-nowrap px-1 rounded text-xs"
          style={{
            fontFamily:  'Orbitron, monospace',
            fontSize:    9,
            color:       style.color,
            background:  'rgba(6,10,16,0.85)',
            letterSpacing: '1px',
          }}
        >
          {track.track_id}
        </div>
      </div>
    </Marker>
  )
}

export default function TacticalMap({ tracks, selectedId, onSelect }: Props) {
  const mapRef = useRef(null)

  const selectedTrack = tracks.find(t => t.track_id === selectedId)

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden border border-vg-border">
      <Map
        ref={mapRef}
        initialViewState={{ latitude: 49, longitude: 15, zoom: 3.8 }}
        style={{ width: '100%', height: '100%' }}
        mapStyle={MAP_STYLE}
        attributionControl={false}
      >
        {/* Historical trail lines via SVG overlay — rendered as markers at history points */}
        {selectedTrack && selectedTrack.hist_lats.map((lat, i) => (
          i < selectedTrack.hist_lats.length - 1 && (
            <Marker key={`trail-${i}`} latitude={lat} longitude={selectedTrack.hist_lons[i]} anchor="center">
              <div
                className="rounded-full"
                style={{
                  width:      5,
                  height:     5,
                  background: CLASS_STYLES[selectedTrack.ai_class]?.color ?? '#38bdf8',
                  opacity:    0.3 + (i / selectedTrack.hist_lats.length) * 0.6,
                }}
              />
            </Marker>
          )
        ))}

        {tracks.map(track => (
          <AircraftMarker
            key={track.track_id}
            track={track}
            selected={track.track_id === selectedId}
            onSelect={onSelect}
          />
        ))}

        {/* Popup on selected track */}
        {selectedTrack && (
          <Popup
            latitude={selectedTrack.latitude}
            longitude={selectedTrack.longitude}
            offset={28}
            closeButton={false}
            closeOnClick={false}
            style={{ background: 'transparent', border: 'none', padding: 0 }}
          >
            <div
              className="rounded-lg px-3 py-2 text-xs"
              style={{
                background:   'rgba(6,10,16,0.95)',
                border:       `1px solid ${CLASS_STYLES[selectedTrack.ai_class]?.color ?? '#38bdf8'}44`,
                fontFamily:   'Space Grotesk, sans-serif',
                color:        '#94a3b8',
                minWidth:     160,
              }}
            >
              <div style={{ fontFamily: 'Orbitron, monospace', fontSize: 10, color: CLASS_STYLES[selectedTrack.ai_class]?.color, letterSpacing: 2, marginBottom: 4 }}>
                {selectedTrack.track_id}
              </div>
              <div>{selectedTrack.altitude_ft.toLocaleString()} ft &nbsp;·&nbsp; {selectedTrack.speed_kts.toFixed(0)} kts</div>
              <div>{selectedTrack.electronic_signature}</div>
            </div>
          </Popup>
        )}
      </Map>

      {/* Legend */}
      <div
        className="absolute bottom-3 left-3 rounded-lg px-3 py-2 flex flex-col gap-1"
        style={{ background: 'rgba(6,10,16,0.9)', border: '1px solid rgba(56,189,248,0.12)', fontFamily: 'Space Grotesk, sans-serif', fontSize: 11 }}
      >
        {Object.entries(CLASS_STYLES).slice(0, 5).map(([cls, s]) => (
          <div key={cls} className="flex items-center gap-2">
            <div className="rounded-full w-2.5 h-2.5 flex-shrink-0" style={{ background: s.color }} />
            <span style={{ color: '#64748b' }}>{cls}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
