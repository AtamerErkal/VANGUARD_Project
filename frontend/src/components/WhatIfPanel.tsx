import { useState } from 'react'
import type { Track, PredictResponse } from '../types'
import { CLASS_STYLES, SIGNATURES, PROFILES, WEATHER_OPTS, THERMAL_OPTS } from '../types'
import { api } from '../api'
import XAIPanel from './XAIPanel'

interface Props { track: Track }

const WEATHER_IRST_FACTOR: Record<string, number> = { Clear: 1.0, Cloudy: 0.55, Rainy: 0.30 }
const WEATHER_ICON: Record<string, string>        = { Clear: '☀️', Cloudy: '⛅', Rainy: '🌧️' }

const IFF_PRESETS = [
  { label: 'Disable IFF',      icon: '📵', sig: 'NO_IFF_RESPONSE',   tip: 'Enemy goes radio-silent' },
  { label: 'Spoof Civilian',   icon: '✈️', sig: 'IFF_MODE_3C',        tip: 'Enemy mimics civil squawk' },
  { label: 'Military Mode',    icon: '🛡️', sig: 'IFF_MODE_5',         tip: 'Friendly identification' },
  { label: 'Active Jamming',   icon: '⚡', sig: 'HOSTILE_JAMMING',    tip: 'Electronic warfare engaged' },
]

export default function WhatIfPanel({ track }: Props) {
  const [open,     setOpen]     = useState(false)
  const [alt,      setAlt]      = useState(track.altitude_ft)
  const [spd,      setSpd]      = useState(track.speed_kts)
  const [rcs,      setRcs]      = useState(track.rcs_m2)
  const [esig,     setEsig]     = useState(track.electronic_signature)
  const [fp,       setFp]       = useState(track.flight_profile)
  const [weather,  setWeather]  = useState(track.weather)
  const [thermal,  setThermal]  = useState(track.thermal_signature)
  const [result,   setResult]   = useState<PredictResponse | null>(null)
  const [loading,  setLoading]  = useState(false)

  const classify = async () => {
    setLoading(true)
    try {
      const res = await api.predict({
        altitude_ft: alt, speed_kts: spd, rcs_m2: rcs,
        latitude: track.latitude, longitude: track.longitude,
        heading: track.heading, weather, thermal_signature: thermal,
        electronic_signature: esig, flight_profile: fp,
      })
      setResult(res)
    } catch (e) { console.error(e) }
    finally { setLoading(false) }
  }

  const origStyle   = CLASS_STYLES[track.ai_class]   ?? CLASS_STYLES['NEUTRAL']
  const resultStyle = result ? (CLASS_STYLES[result.classification] ?? CLASS_STYLES['NEUTRAL']) : null
  const changed     = result && result.classification !== track.ai_class

  // Live IRST preview for current weather selection
  const irstBaseConf = track.sensor_votes?.irst?.base_conf ?? track.sensor_votes?.irst?.conf ?? 0.5
  const irstLive     = +(irstBaseConf * (WEATHER_IRST_FACTOR[weather] ?? 1)).toFixed(2)

  return (
    <div className="rounded-xl overflow-hidden" style={{ border: '1px solid rgba(56,189,248,0.1)' }}>
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between px-4 py-3 text-xs font-semibold tracking-widest uppercase transition-colors"
        style={{ background: 'rgba(12,18,30,0.85)', color: '#38bdf888', fontFamily: 'Orbitron, monospace' }}
      >
        <span>🔬 What-If Analysis</span>
        <span style={{ color: '#475569' }}>{open ? '▲' : '▼'}</span>
      </button>

      {open && (
        <div className="px-4 py-4 space-y-5" style={{ background: 'rgba(8,12,20,0.95)' }}>

          {/* ── IFF Scenario Presets ── */}
          <div className="space-y-2">
            <p className="text-xs tracking-widest uppercase" style={{ color: '#64748b', fontFamily: 'Orbitron, monospace', fontSize: 9 }}>
              IFF Scenario Presets
            </p>
            <div className="grid grid-cols-2 gap-1.5">
              {IFF_PRESETS.map(preset => (
                <button
                  key={preset.sig}
                  onClick={() => setEsig(preset.sig)}
                  title={preset.tip}
                  className="text-left px-2.5 py-2 rounded-lg text-xs transition-all hover:brightness-125"
                  style={{
                    background:  esig === preset.sig ? 'rgba(56,189,248,0.15)' : 'rgba(12,18,30,0.8)',
                    border:      `1px solid ${esig === preset.sig ? 'rgba(56,189,248,0.5)' : 'rgba(56,189,248,0.1)'}`,
                    color:       esig === preset.sig ? '#38bdf8' : '#64748b',
                    fontFamily:  'Space Grotesk, sans-serif',
                  }}
                >
                  <span className="mr-1">{preset.icon}</span>{preset.label}
                </button>
              ))}
            </div>
          </div>

          {/* ── Weather + IRST impact ── */}
          <div className="space-y-2">
            <p className="text-xs tracking-widest uppercase" style={{ color: '#64748b', fontFamily: 'Orbitron, monospace', fontSize: 9 }}>
              Weather — IRST Impact
            </p>
            <div className="grid grid-cols-3 gap-1.5">
              {WEATHER_OPTS.map(w => {
                const factor = WEATHER_IRST_FACTOR[w]
                const conf   = +(irstBaseConf * factor).toFixed(2)
                const active = weather === w
                return (
                  <button
                    key={w}
                    onClick={() => setWeather(w)}
                    className="rounded-lg px-2 py-2 text-center transition-all hover:brightness-125"
                    style={{
                      background: active ? 'rgba(56,189,248,0.12)' : 'rgba(12,18,30,0.8)',
                      border:     `1px solid ${active ? 'rgba(56,189,248,0.4)' : 'rgba(56,189,248,0.08)'}`,
                    }}
                  >
                    <div className="text-base mb-0.5">{WEATHER_ICON[w]}</div>
                    <div className="text-xs font-semibold" style={{ color: active ? '#38bdf8' : '#64748b' }}>{w}</div>
                    <div className="text-xs mt-1" style={{ color: '#475569', fontSize: 10 }}>IRST</div>
                    <div className="h-1 rounded-full mt-0.5 mx-1" style={{ background: 'rgba(255,255,255,0.06)' }}>
                      <div className="h-full rounded-full" style={{ width: `${conf * 100}%`, background: factor === 1 ? '#4ade80' : factor >= 0.5 ? '#fbbf24' : '#f87171' }} />
                    </div>
                    <div className="text-xs mt-0.5" style={{ color: factor === 1 ? '#4ade80' : factor >= 0.5 ? '#fbbf24' : '#f87171', fontSize: 10 }}>
                      {(conf * 100).toFixed(0)}%
                    </div>
                  </button>
                )
              })}
            </div>
            {weather !== track.weather && (
              <p className="text-xs" style={{ color: '#f59e0b' }}>
                ⚠ Weather changed: IRST {(irstBaseConf * 100).toFixed(0)}% → {(irstLive * 100).toFixed(0)}%
                {weather === 'Rainy' ? ' — severely degraded' : weather === 'Cloudy' ? ' — moderately degraded' : ' — restored'}
              </p>
            )}
          </div>

          {/* ── Kinematic sliders ── */}
          <div className="space-y-2">
            <p className="text-xs tracking-widest uppercase" style={{ color: '#64748b', fontFamily: 'Orbitron, monospace', fontSize: 9 }}>
              Kinematic Parameters
            </p>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: `Altitude: ${alt.toLocaleString()} ft`, min: 0,   max: 60000, step: 500, val: alt, set: setAlt },
                { label: `Speed: ${spd.toFixed(0)} kts`,         min: 0,   max: 1000,  step: 10,  val: spd, set: setSpd },
                { label: `RCS: ${rcs.toFixed(1)} m²`,            min: 0.1, max: 100,   step: 0.5, val: rcs, set: setRcs },
              ].map(({ label, min, max, step, val, set }) => (
                <label key={label} className="space-y-1 col-span-1">
                  <span className="text-xs" style={{ color: '#64748b' }}>{label}</span>
                  <input type="range" min={min} max={max} step={step} value={val}
                    onChange={e => set(+e.target.value)} className="w-full accent-sky-400 h-1" />
                </label>
              ))}

              <label className="space-y-1">
                <span className="text-xs" style={{ color: '#64748b' }}>Thermal Signature</span>
                <select value={thermal} onChange={e => setThermal(e.target.value)}
                  className="w-full text-xs py-1.5 px-2 rounded-lg outline-none"
                  style={{ background: 'rgba(12,18,30,0.9)', border: '1px solid rgba(56,189,248,0.2)', color: '#94a3b8' }}>
                  {THERMAL_OPTS.map(t => <option key={t} value={t} style={{ background: '#0d1117' }}>{t}</option>)}
                </select>
              </label>
            </div>
          </div>

          {/* ── Flight profile ── */}
          <div className="space-y-1">
            <p className="text-xs tracking-widest uppercase" style={{ color: '#64748b', fontFamily: 'Orbitron, monospace', fontSize: 9 }}>
              Flight Profile
            </p>
            <div className="grid grid-cols-2 gap-1.5">
              {PROFILES.map(p => (
                <button key={p} onClick={() => setFp(p)}
                  className="text-xs py-1.5 px-2 rounded-lg transition-all hover:brightness-125"
                  style={{
                    background: fp === p ? 'rgba(167,139,250,0.15)' : 'rgba(12,18,30,0.8)',
                    border:     `1px solid ${fp === p ? 'rgba(167,139,250,0.5)' : 'rgba(56,189,248,0.08)'}`,
                    color:      fp === p ? '#a78bfa' : '#64748b',
                  }}>
                  {p.replace(/_/g, ' ')}
                </button>
              ))}
            </div>
          </div>

          {/* ── Classify ── */}
          <button
            onClick={classify} disabled={loading}
            className="w-full py-2.5 text-xs font-bold rounded-lg tracking-widest uppercase transition-all hover:brightness-110 disabled:opacity-50"
            style={{ background: 'linear-gradient(180deg,#38bdf8,#0284c7)', color: 'white', fontFamily: 'Orbitron, monospace' }}
          >
            {loading ? 'CLASSIFYING...' : '🔄 RE-CLASSIFY'}
          </button>

          {/* ── Result diff ── */}
          {result && resultStyle && (
            <div className="space-y-3 result-reveal">
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'ORIGINAL', cls: track.ai_class, conf: track.ai_conf, s: origStyle, glow: false },
                  { label: changed ? 'MODIFIED ← CHANGED' : 'MODIFIED', cls: result.classification, conf: result.confidence, s: resultStyle, glow: !!changed },
                ].map(({ label, cls, conf, s, glow }) => (
                  <div key={label} className="rounded-xl py-3 px-3 text-center"
                    style={{
                      background: s.bg,
                      border:     `1px solid ${s.color}${glow ? '88' : '33'}`,
                      boxShadow:  glow ? `0 0 20px ${s.color}33` : 'none',
                      fontFamily: 'Orbitron, monospace',
                    }}>
                    <div style={{ color: '#94a3b8', fontSize: 10, letterSpacing: 2, marginBottom: 4 }}>{label}</div>
                    <div style={{ color: s.color, letterSpacing: 2, fontSize: 14, fontWeight: 900 }}>{s.icon} {cls}</div>
                    <div style={{ color: '#cbd5e1', fontSize: 13, marginTop: 4 }}>{(conf * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>

              {/* XAI for re-classified result */}
              {result.xai && result.xai.length > 0 && (
                <XAIPanel xai={result.xai} aiClass={result.classification} />
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
