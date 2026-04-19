import { useState } from 'react'
import type { Track, PredictResponse } from '../types'
import { CLASS_STYLES } from '../types'
import { api } from '../api'

interface Props { track: Track }

const SIGNATURES = ['IFF_MODE_5', 'IFF_MODE_3C', 'NO_IFF_RESPONSE', 'HOSTILE_JAMMING', 'UNKNOWN_EMISSION']
const PROFILES   = ['STABLE_CRUISE', 'AGGRESSIVE_MANEUVERS', 'LOW_ALTITUDE_FLYING', 'CLIMBING']

export default function WhatIfPanel({ track }: Props) {
  const [open,    setOpen]    = useState(false)
  const [alt,     setAlt]     = useState(track.altitude_ft)
  const [spd,     setSpd]     = useState(track.speed_kts)
  const [rcs,     setRcs]     = useState(track.rcs_m2)
  const [esig,    setEsig]    = useState(track.electronic_signature)
  const [fp,      setFp]      = useState(track.flight_profile)
  const [result,  setResult]  = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const classify = async () => {
    setLoading(true)
    try {
      const res = await api.predict({
        altitude_ft: alt, speed_kts: spd, rcs_m2: rcs,
        latitude: track.latitude, longitude: track.longitude,
        heading: track.heading, weather: track.weather,
        electronic_signature: esig, flight_profile: fp,
        thermal_signature: track.thermal_signature,
      })
      setResult(res)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  const origStyle   = CLASS_STYLES[track.ai_class]   ?? CLASS_STYLES['NEUTRAL']
  const resultStyle = result ? (CLASS_STYLES[result.classification] ?? CLASS_STYLES['NEUTRAL']) : null
  const changed     = result && result.classification !== track.ai_class

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid rgba(56,189,248,0.1)' }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold tracking-widest uppercase transition-colors duration-150"
        style={{ background: 'rgba(12,18,30,0.85)', color: '#38bdf888', fontFamily: 'Orbitron, monospace' }}
      >
        <span>🔬 What-If Analysis</span>
        <span style={{ color: '#475569' }}>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="px-4 py-4 space-y-4" style={{ background: 'rgba(8,12,20,0.9)' }}>
          <p className="text-xs" style={{ color: '#64748b' }}>Adjust parameters and re-classify to see model response.</p>

          <div className="grid grid-cols-2 gap-3">
            <label className="space-y-1">
              <span className="text-xs" style={{ color: '#64748b' }}>Altitude (ft): {alt.toLocaleString()}</span>
              <input type="range" min={0} max={60000} step={500} value={alt}
                onChange={e => setAlt(+e.target.value)} className="w-full accent-sky-400" />
            </label>
            <label className="space-y-1">
              <span className="text-xs" style={{ color: '#64748b' }}>Speed (kts): {spd.toFixed(0)}</span>
              <input type="range" min={0} max={1000} step={10} value={spd}
                onChange={e => setSpd(+e.target.value)} className="w-full accent-sky-400" />
            </label>
            <label className="space-y-1">
              <span className="text-xs" style={{ color: '#64748b' }}>RCS (m²): {rcs.toFixed(1)}</span>
              <input type="range" min={0.1} max={100} step={0.5} value={rcs}
                onChange={e => setRcs(+e.target.value)} className="w-full accent-sky-400" />
            </label>
            <label className="space-y-1">
              <span className="text-xs" style={{ color: '#64748b' }}>Electronic Signature</span>
              <select value={esig} onChange={e => setEsig(e.target.value)}
                className="w-full text-xs py-1.5 px-2 rounded-lg outline-none"
                style={{ background: 'rgba(12,18,30,0.9)', border: '1px solid rgba(56,189,248,0.2)', color: '#94a3b8' }}>
                {SIGNATURES.map(s => <option key={s} value={s} style={{ background: '#0d1117' }}>{s}</option>)}
              </select>
            </label>
            <label className="space-y-1 col-span-2">
              <span className="text-xs" style={{ color: '#64748b' }}>Flight Profile</span>
              <select value={fp} onChange={e => setFp(e.target.value)}
                className="w-full text-xs py-1.5 px-2 rounded-lg outline-none"
                style={{ background: 'rgba(12,18,30,0.9)', border: '1px solid rgba(56,189,248,0.2)', color: '#94a3b8' }}>
                {PROFILES.map(p => <option key={p} value={p} style={{ background: '#0d1117' }}>{p}</option>)}
              </select>
            </label>
          </div>

          <button
            onClick={classify}
            disabled={loading}
            className="w-full py-2.5 text-xs font-bold rounded-lg tracking-widest uppercase transition-all duration-150 hover:brightness-110 disabled:opacity-50"
            style={{ background: 'linear-gradient(180deg, #38bdf8, #0284c7)', color: 'white', fontFamily: 'Orbitron, monospace' }}
          >
            {loading ? 'CLASSIFYING...' : '🔄 RE-CLASSIFY'}
          </button>

          {result && resultStyle && (
            <div className="grid grid-cols-2 gap-3 result-reveal">
              {[
                { label: 'ORIGINAL', cls: track.ai_class, conf: track.ai_conf, style: origStyle, highlight: false },
                { label: changed ? 'MODIFIED ← CHANGED' : 'MODIFIED', cls: result.classification, conf: result.confidence, style: resultStyle, highlight: !!changed },
              ].map(({ label, cls, conf, style, highlight }) => (
                <div key={label} className="rounded-xl py-3 px-3 text-center"
                  style={{
                    background:  style.bg,
                    border:      `1px solid ${style.color}${highlight ? '88' : '33'}`,
                    boxShadow:   highlight ? `0 0 20px ${style.color}33` : 'none',
                    fontFamily:  'Orbitron, monospace',
                  }}>
                  <div className="text-xs mb-1 tracking-widest" style={{ color: '#475569', fontSize: 9 }}>{label}</div>
                  <div className="font-black" style={{ color: style.color, letterSpacing: 2, fontSize: 13 }}>
                    {style.icon} {cls}
                  </div>
                  <div className="text-xs mt-1" style={{ color: '#475569' }}>{(conf * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
