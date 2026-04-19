export interface SensorVote {
  label:   string
  icon:    string
  vote:    string
  conf:    number
  reading: string
}

export interface FusionResult {
  best:    string
  probs:   Record<string, number>
  weights: Record<string, number>
}

export interface Anomaly {
  title: string
  desc:  string
}

export interface Track {
  track_id:             string
  latitude:             number
  longitude:            number
  altitude_ft:          number
  speed_kts:            number
  rcs_m2:               number
  heading:              number
  weather:              string
  electronic_signature: string
  flight_profile:       string
  thermal_signature:    string
  ai_class:             string
  ai_conf:              number
  ai_probs:             Record<string, number>
  sensor_votes:         Record<string, SensorVote>
  fusion:               FusionResult
  anomalies:            Anomaly[]
  hist_lats:            number[]
  hist_lons:            number[]
  hist_alts:            number[]
}

export interface PredictRequest {
  altitude_ft:          number
  speed_kts:            number
  rcs_m2:               number
  latitude:             number
  longitude:            number
  heading:              number
  electronic_signature: string
  flight_profile:       string
  weather:              string
  thermal_signature:    string
}

export interface PredictResponse {
  classification: string
  confidence:     number
  probabilities:  Record<string, number>
}

export type ApprovalAction = 'approved' | 'override'

export interface ApprovalState {
  action:          ApprovalAction
  override_class?: string
}

export const CLASS_STYLES: Record<string, { color: string; bg: string; icon: string }> = {
  'HOSTILE':        { color: '#ef4444', bg: '#450a0a', icon: '🚨' },
  'SUSPECT':        { color: '#f59e0b', bg: '#451a03', icon: '⚠️' },
  'FRIEND':         { color: '#22c55e', bg: '#064e3b', icon: '🛡️' },
  'ASSUMED FRIEND': { color: '#22c55e', bg: '#064e3b', icon: '🤝' },
  'NEUTRAL':        { color: '#94a3b8', bg: '#1e293b', icon: '🏳️' },
  'CIVILIAN':       { color: '#38bdf8', bg: '#0c4a6e', icon: '✈️' },
}

export const SENSOR_ORDER = ['radar', 'esm', 'irst', 'iff'] as const
