import React, { useState } from 'react'
import { useParams, Link } from 'react-router-dom'
import { CLASS_STYLES, ESM_SIGS, IFF_MODES, PROFILES, WEATHER_OPTS, THERMAL_OPTS } from '../types'
import type { SimTrack } from '../sim-types'

const BASE = import.meta.env.VITE_API_URL ?? ''

const RCS_PRESETS = [
  { label: 'Stealth',      desc: '~0.02 m²',  value: 0.02 },
  { label: 'Small Ftr',   desc: '~0.5 m²',   value: 0.5  },
  { label: 'Med Ftr',     desc: '~2.0 m²',   value: 2.0  },
  { label: 'Bomber',      desc: '~8.0 m²',   value: 8.0  },
  { label: 'Transport',   desc: '~20 m²',    value: 20.0 },
]

function Slider({
  label, unit, min, max, step, value, onChange,
  display,
}: {
  label: string; unit: string; min: number; max: number; step: number
  value: number; onChange: (v: number) => void
  display?: string
}) {
  return (
    <label className="flex flex-col gap-1.5">
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="font-mono text-cyan-300 font-semibold">{display ?? value} {unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(+e.target.value)}
        className="w-full h-1.5 rounded-full accent-cyan-500 cursor-pointer"
      />
    </label>
  )
}

function ToggleGroup<T extends string>({
  label, options, value, onChange, accent,
}: {
  label: string; options: readonly T[]; value: T
  onChange: (v: T) => void; accent: string
}) {
  return (
    <label className="flex flex-col gap-2">
      <span className="text-xs text-slate-400">{label}</span>
      <div className="flex flex-wrap gap-1.5">
        {options.map(o => (
          <button key={o} type="button" onClick={() => onChange(o)}
            className={`px-2.5 py-1 rounded-lg text-xs font-mono transition-colors border ${
              value === o
                ? `${accent} text-white border-transparent`
                : 'bg-slate-800/80 text-slate-400 border-slate-700/60 hover:bg-slate-700'
            }`}
          >{o.replace(/_/g,' ')}</button>
        ))}
      </div>
    </label>
  )
}

export default function SimSubmit() {
  const { sessionId } = useParams<{ sessionId: string }>()

  // ── Form state ─────────────────────────────────────────────────────────────
  const [altitude_ft,       setAlt]      = useState(25000)
  const [speed_kts,         setSpd]      = useState(450)
  const [rcs_m2,            setRcs]      = useState(2.0)
  const [heading,           setHdg]      = useState(270)
  const [esm_signature,     setEsm]      = useState<string>('CLEAN')
  const [iff_mode,          setIff]      = useState<string>('NO_RESPONSE')
  const [flight_profile,    setFp]       = useState<string>('STABLE_CRUISE')
  const [weather,           setWeather]  = useState<string>('Clear')
  const [thermal_signature, setThermal]  = useState<string>('Medium')

  const [result,  setResult]  = useState<SimTrack | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  // ── Submit ─────────────────────────────────────────────────────────────────
  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setLoading(true); setError(null); setResult(null)
    try {
      const body = { altitude_ft, speed_kts, rcs_m2, heading,
                     esm_signature, iff_mode, flight_profile, weather, thermal_signature }
      const res = await fetch(`${BASE}/sim/${sessionId}/submit`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(body),
      })
      if (!res.ok) {
        const txt = await res.text().catch(() => '')
        throw new Error(`HTTP ${res.status}${txt ? ': ' + txt : ''}`)
      }
      setResult(await res.json())
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const s = result ? CLASS_STYLES[result.ai_class] : null

  return (
    <div className="min-h-screen bg-[#060d19] text-white px-4 py-6 md:py-10">
      <div className="max-w-lg mx-auto space-y-6">

        {/* Header */}
        <div>
          <p className="text-xs text-slate-500 tracking-widest mb-1">VANGUARD AI · TACTICAL SIM</p>
          <h1 className="text-xl font-bold text-cyan-400 tracking-wider">📡 TRACK SUBMISSION</h1>
          <div className="flex items-center gap-2 mt-1 text-sm text-slate-400">
            <span>Session:</span>
            <span className="font-mono font-bold text-yellow-300 tracking-widest">{sessionId}</span>
          </div>
          <p className="text-xs text-slate-600 mt-1">
            Configure a contact's parameters and submit — the AI will classify it and
            it will appear on the operator's tactical display.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">

          {/* ── Kinematic ─────────────────────────────────────────────── */}
          <section className="bg-slate-900/60 border border-slate-800/70 rounded-2xl p-4 space-y-4">
            <h3 className="text-[10px] font-bold text-cyan-500 tracking-[0.2em]">📊 KINEMATIC DATA</h3>

            <Slider label="Altitude"   unit="ft"  min={100}  max={50000} step={100} value={altitude_ft} onChange={setAlt} display={altitude_ft.toLocaleString()} />
            <Slider label="Speed"      unit="kts" min={50}   max={1800}  step={10}  value={speed_kts}   onChange={setSpd} />
            <Slider label="Heading"    unit="°"   min={0}    max={359}   step={1}   value={heading}     onChange={setHdg} />

            {/* RCS presets */}
            <label className="flex flex-col gap-2">
              <span className="text-xs text-slate-400">Radar Cross Section (RCS)</span>
              <div className="grid grid-cols-5 gap-1.5">
                {RCS_PRESETS.map(p => (
                  <button key={p.value} type="button"
                    onClick={() => setRcs(p.value)}
                    className={`py-2 rounded-xl text-center transition-colors border ${
                      rcs_m2 === p.value
                        ? 'bg-cyan-800 border-cyan-600 text-white'
                        : 'bg-slate-800/70 border-slate-700/50 text-slate-400 hover:bg-slate-700'
                    }`}
                  >
                    <div className="text-[10px] font-bold">{p.label}</div>
                    <div className="text-[9px] text-slate-500">{p.desc}</div>
                  </button>
                ))}
              </div>
            </label>

            <ToggleGroup label="Flight Profile" options={PROFILES} value={flight_profile as any} onChange={setFp} accent="bg-cyan-800" />
          </section>

          {/* ── Electronic Warfare ────────────────────────────────────── */}
          <section className="bg-slate-900/60 border border-slate-800/70 rounded-2xl p-4 space-y-4">
            <h3 className="text-[10px] font-bold text-orange-400 tracking-[0.2em]">📻 ELECTRONIC WARFARE</h3>

            <ToggleGroup
              label="ESM Signature — Electronic Support Measures (passive — what the aircraft emits)"
              options={ESM_SIGS} value={esm_signature as any} onChange={setEsm}
              accent="bg-orange-800"
            />
            <ToggleGroup
              label="IFF Mode — Identification Friend or Foe (active transponder response)"
              options={IFF_MODES} value={iff_mode as any} onChange={setIff}
              accent="bg-blue-800"
            />
          </section>

          {/* ── Environment ───────────────────────────────────────────── */}
          <section className="bg-slate-900/60 border border-slate-800/70 rounded-2xl p-4 space-y-4">
            <h3 className="text-[10px] font-bold text-green-400 tracking-[0.2em]">🌤 ENVIRONMENT</h3>

            <ToggleGroup label="Weather"           options={WEATHER_OPTS} value={weather as any}           onChange={setWeather}  accent="bg-green-800" />
            <ToggleGroup label="Thermal Signature" options={THERMAL_OPTS} value={thermal_signature as any} onChange={setThermal}  accent="bg-green-800" />
          </section>

          {/* Submit button */}
          <button type="submit" disabled={loading}
            className="w-full bg-cyan-700 hover:bg-cyan-600 active:scale-[0.98] disabled:opacity-50
                       disabled:cursor-not-allowed text-white font-bold py-3.5 rounded-2xl
                       transition-all tracking-widest text-sm"
          >
            {loading ? '⏳  CLASSIFYING…' : '🚀  SUBMIT TRACK'}
          </button>
        </form>

        {/* Error */}
        {error && (
          <div className="bg-red-950/50 border border-red-800/60 rounded-xl p-4 text-sm text-red-400">
            ⚠ {error}
          </div>
        )}

        {/* Result */}
        {result && s && (
          <div className="rounded-2xl border-2 p-5 space-y-4"
            style={{ borderColor: s.color, backgroundColor: s.bg + 'bb' }}
          >
            {/* Classification headline */}
            <div className="flex items-center gap-4">
              <span className="text-5xl">{s.icon}</span>
              <div className="flex-1">
                <p className="text-[10px] text-slate-400 tracking-[0.2em]">AI CLASSIFICATION</p>
                <p className="text-2xl font-bold font-mono tracking-wider" style={{ color: s.color }}>
                  {result.ai_class}
                </p>
              </div>
              <div className="text-right">
                <p className="text-[10px] text-slate-400 tracking-wider">CONFIDENCE</p>
                <p className="text-3xl font-bold font-mono" style={{ color: s.color }}>
                  {Math.round(result.ai_conf * 100)}%
                </p>
              </div>
            </div>

            {/* Sensor vote breakdown */}
            {result.sensor_votes && (
              <div className="space-y-1.5">
                <p className="text-[9px] text-slate-500 tracking-[0.2em] font-bold">SENSOR VOTES</p>
                {Object.values(result.sensor_votes).map(sv => (
                  <div key={sv.label} className="flex items-center gap-2 text-xs">
                    <span className="text-base">{sv.icon}</span>
                    <span className="text-slate-400 w-10">{sv.label}</span>
                    <span className="text-slate-300 flex-1 font-mono text-[10px]">{sv.reading}</span>
                    <span className="font-bold font-mono text-[10px]" style={{ color: s.color }}>
                      {sv.vote} {Math.round(sv.conf * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            )}

            {/* Track metadata */}
            <div className="flex items-center justify-between text-[10px] text-slate-500 pt-1 border-t border-white/10">
              <span>Track ID: <span className="font-mono text-yellow-300">{result.track_id}</span></span>
              <span>Map: ({result.pos.x}, {result.pos.y})</span>
              <span>{result.submitted_at}</span>
            </div>

            {/* Submit another */}
            <button
              onClick={() => { setResult(null); setError(null) }}
              className="w-full text-xs py-2 border border-slate-700 hover:border-slate-500
                         text-slate-500 hover:text-white rounded-xl transition-colors"
            >
              Submit another track →
            </button>
          </div>
        )}

        <div className="text-center pb-4">
          <Link to="/sim" className="text-slate-700 hover:text-slate-500 text-xs transition-colors">
            ← Simulation Home
          </Link>
        </div>
      </div>
    </div>
  )
}
