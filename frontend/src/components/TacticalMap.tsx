import { useRef, useEffect, useState } from 'react'
import Map, { Marker, Popup } from 'react-map-gl/maplibre'
import 'maplibre-gl/dist/maplibre-gl.css'
import type { Track, ApprovalState } from '../types'
import { CLASS_STYLES } from '../types'

const MAP_STYLE = 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
const STEP      = 0.0018   // degrees per tick
const TICK_MS   = 900      // ms between moves

interface Pos { lat: number; lon: number }

interface Props {
  tracks:     Track[]
  selectedId: string | null
  approvals:  Record<string, ApprovalState>
  onSelect:   (id: string) => void
}

function AircraftMarker({ track, pos, selected, approved, onSelect }: {
  track:    Track
  pos:      Pos
  selected: boolean
  approved: boolean
  onSelect: (id: string) => void
}) {
  const style     = CLASS_STYLES[track.ai_class] ?? CLASS_STYLES['NEUTRAL']
  const isHostile = ['HOSTILE', 'SUSPECT'].includes(track.ai_class)
  const size      = selected ? 40 : 32

  return (
    <Marker latitude={pos.lat} longitude={pos.lon} anchor="center"
      onClick={e => { e.originalEvent.stopPropagation(); onSelect(track.track_id) }}>
      <div className="cursor-pointer" style={{ position: 'relative', width: size + 16, height: size + 24 }}>

        {/* Pulse ring — hostile/suspect only */}
        {isHostile && !approved && (
          <span className="animate-ping"
                style={{
                  position: 'absolute', top: 8, left: 8,
                  width: size, height: size,
                  borderRadius: '50%',
                  background: style.color,
                  opacity: 0.3,
                }} />
        )}

        {/* Main marker — circle → square when approved */}
        <div style={{
          position:     'absolute',
          top:          8, left: 8,
          width:        size,
          height:       size,
          background:   style.bg,
          border:       `2px solid ${style.color}`,
          borderRadius: approved ? '4px' : '50%',
          transform:    `${selected ? 'scale(1.15)' : 'scale(1)'} ${approved ? 'rotate(0deg)' : ''}`,
          transition:   'border-radius 0.4s ease, transform 0.2s ease',
          boxShadow:    selected
            ? `0 0 20px ${style.color}99, 0 0 40px ${style.color}33`
            : `0 0 8px ${style.color}55`,
          display:      'flex',
          alignItems:   'center',
          justifyContent: 'center',
          fontSize:     selected ? 16 : 13,
        }}>
          {approved ? '✓' : style.icon}
        </div>

        {/* Heading tick — longer for approved tracks */}
        <div style={{
          position:        'absolute',
          top:             8 - (approved ? 26 : 20),
          left:            8 + size / 2 - 1,
          width:           approved ? 3 : 2,
          height:          approved ? 26 : 20,
          background:      approved
            ? `linear-gradient(to top, ${style.color}, ${style.color}cc, transparent)`
            : `linear-gradient(to top, ${style.color}, ${style.color}88, transparent)`,
          transformOrigin: `1px ${(approved ? 26 : 20) + size / 2}px`,
          transform:       `rotate(${track.heading}deg)`,
          borderRadius:    '2px 2px 0 0',
        }} />

        {/* Track ID label */}
        <div style={{
          position:      'absolute',
          top:           size + 12,
          left:          '50%',
          transform:     'translateX(-50%)',
          whiteSpace:    'nowrap',
          fontFamily:    'Orbitron, monospace',
          fontSize:      9,
          letterSpacing: 1,
          color:         style.color,
          background:    'rgba(6,10,16,0.88)',
          padding:       '1px 4px',
          borderRadius:  3,
        }}>
          {track.track_id}
        </div>
      </div>
    </Marker>
  )
}

export default function TacticalMap({ tracks, selectedId, approvals, onSelect }: Props) {
  const mapRef = useRef(null)

  // Animated positions
  const [positions, setPositions] = useState<Record<string, Pos>>({})

  useEffect(() => {
    if (!tracks.length) return
    const init: Record<string, Pos> = {}
    tracks.forEach(t => { init[t.track_id] = { lat: t.latitude, lon: t.longitude } })
    setPositions(init)

    const id = setInterval(() => {
      setPositions(prev => {
        const next = { ...prev }
        tracks.forEach(t => {
          const p   = prev[t.track_id] ?? { lat: t.latitude, lon: t.longitude }
          const ang = (t.heading * Math.PI) / 180
          next[t.track_id] = {
            lat: p.lat + STEP * Math.cos(ang),
            lon: p.lon + STEP * Math.sin(ang),
          }
        })
        return next
      })
    }, TICK_MS)

    return () => clearInterval(id)
  }, [tracks])

  const selTrack = tracks.find(t => t.track_id === selectedId)
  const selPos   = selectedId ? (positions[selectedId] ?? null) : null

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden"
         style={{ border: '1px solid rgba(56,189,248,0.15)' }}>
      <Map
        ref={mapRef}
        initialViewState={{ latitude: 49, longitude: 15, zoom: 3.8 }}
        style={{ width: '100%', height: '100%' }}
        mapStyle={MAP_STYLE}
        attributionControl={false}
      >
        {/* Trail dots for selected track */}
        {selTrack && selTrack.hist_lats.map((lat, i) =>
          i < selTrack.hist_lats.length - 1 ? (
            <Marker key={`trail-${i}`} latitude={lat} longitude={selTrack.hist_lons[i]} anchor="center">
              <div style={{
                width:        i > selTrack.hist_lats.length - 4 ? 7 : 5,
                height:       i > selTrack.hist_lats.length - 4 ? 7 : 5,
                borderRadius: '50%',
                background:   CLASS_STYLES[selTrack.ai_class]?.color ?? '#38bdf8',
                opacity:      0.3 + (i / selTrack.hist_lats.length) * 0.7,
                boxShadow:    i > selTrack.hist_lats.length - 3
                  ? `0 0 6px ${CLASS_STYLES[selTrack.ai_class]?.color ?? '#38bdf8'}88`
                  : 'none',
              }} />
            </Marker>
          ) : null
        )}

        {/* Aircraft markers */}
        {tracks.map(track => {
          const pos = positions[track.track_id]
          if (!pos) return null
          const approval = approvals[track.track_id]
          const approved = approval?.action === 'approved'
          return (
            <AircraftMarker
              key={track.track_id}
              track={track}
              pos={pos}
              selected={track.track_id === selectedId}
              approved={approved}
              onSelect={onSelect}
            />
          )
        })}

        {/* Modern popup for selected track */}
        {selTrack && selPos && (() => {
          const s = CLASS_STYLES[selTrack.ai_class] ?? CLASS_STYLES['NEUTRAL']
          return (
            <Popup
              latitude={selPos.lat}
              longitude={selPos.lon}
              offset={30}
              closeButton={false}
              closeOnClick={false}
              style={{ background: 'transparent', border: 'none', padding: 0, filter: 'none' }}
            >
              <div style={{
                background:     'linear-gradient(135deg, rgba(8,14,26,0.97) 0%, rgba(14,22,40,0.95) 100%)',
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                border:         `1px solid ${s.color}33`,
                borderLeft:     `3px solid ${s.color}`,
                borderRadius:   10,
                padding:        '10px 14px',
                minWidth:       200,
                boxShadow:      `0 8px 32px rgba(0,0,0,0.6), 0 0 24px ${s.color}12`,
                fontFamily:     'Space Grotesk, sans-serif',
              }}>
                {/* Header row */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <span style={{ fontFamily: 'Orbitron, monospace', fontSize: 11, color: s.color, letterSpacing: 3 }}>
                    {selTrack.track_id}
                  </span>
                  <span style={{ fontFamily: 'Orbitron, monospace', fontSize: 10, background: s.bg, color: s.color, padding: '2px 8px', borderRadius: 4, letterSpacing: 1 }}>
                    {s.icon} {selTrack.ai_class}
                  </span>
                </div>

                {/* Data rows */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '3px 12px', fontSize: 12 }}>
                  {[
                    ['Alt',     `${selTrack.altitude_ft.toLocaleString()} ft`],
                    ['Speed',   `${selTrack.speed_kts.toFixed(0)} kts`],
                    ['RCS',     `${selTrack.rcs_m2.toFixed(1)} m²`],
                    ['Heading', `${selTrack.heading}°`],
                  ].map(([label, val]) => (
                    <div key={label}>
                      <span style={{ color: '#475569', fontSize: 10 }}>{label} </span>
                      <span style={{ color: '#e2e8f0', fontWeight: 600 }}>{val}</span>
                    </div>
                  ))}
                </div>

                {/* IFF row */}
                <div style={{ marginTop: 7, paddingTop: 7, borderTop: '1px solid rgba(255,255,255,0.06)', fontSize: 11, color: '#94a3b8' }}>
                  ESM: {selTrack.esm_signature} · IFF: {selTrack.iff_mode}
                  <span style={{ marginLeft: 8, color: '#475569' }}>
                    {selTrack.weather} · {selTrack.thermal_signature}
                  </span>
                </div>

                {/* Confidence bar */}
                <div style={{ marginTop: 7, display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ flex: 1, height: 3, background: 'rgba(255,255,255,0.06)', borderRadius: 2 }}>
                    <div style={{ width: `${selTrack.ai_conf * 100}%`, height: '100%', background: s.color, borderRadius: 2 }} />
                  </div>
                  <span style={{ fontFamily: 'Orbitron, monospace', fontSize: 10, color: s.color }}>
                    {(selTrack.ai_conf * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            </Popup>
          )
        })()}
      </Map>

      {/* Legend */}
      <div style={{
        position: 'absolute', bottom: 12, left: 12,
        background: 'rgba(6,10,16,0.92)', border: '1px solid rgba(56,189,248,0.12)',
        borderRadius: 10, padding: '8px 12px',
        fontFamily: 'Space Grotesk, sans-serif',
        display: 'flex', flexDirection: 'column', gap: 5,
      }}>
        {Object.entries(CLASS_STYLES).slice(0, 5).map(([cls, s]) => (
          <div key={cls} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: s.color, flexShrink: 0 }} />
            <span style={{ color: '#64748b', fontSize: 11 }}>{cls}</span>
          </div>
        ))}
        <div style={{ marginTop: 4, paddingTop: 4, borderTop: '1px solid rgba(255,255,255,0.06)', display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ width: 8, height: 8, borderRadius: 2, background: '#22c55e', flexShrink: 0 }} />
          <span style={{ color: '#64748b', fontSize: 11 }}>APPROVED</span>
        </div>
      </div>
    </div>
  )
}
