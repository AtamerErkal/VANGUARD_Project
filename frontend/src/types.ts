export interface SensorVote {
  label:          string
  icon:           string
  vote:           string
  conf:           number
  reading:        string
  weather_factor?: number
  base_conf?:      number
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

export interface XAIItem {
  feature:    string
  label:      string
  group:      string
  value:      string
  importance: number
  direction:  'supporting' | 'conflicting' | 'neutral'
  delta:      number
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
  xai:                  XAIItem[]
  weather_impact:       Record<string, number>
  hist_lats:            number[]
  hist_lons:            number[]
  hist_alts:            number[]
  hist_speeds:          number[]
  hist_headings:        number[]
  hist_timestamps:      string[]
}

export interface PerClassMetric {
  f1:        number
  precision: number
  recall:    number
  support:   number
}

export interface ModelStats {
  accuracy:         number
  f1_macro:         number
  f1_weighted:      number
  test_size:        number
  train_size:       number
  classes:          string[]
  per_class:        Record<string, PerClassMetric>
  confusion_matrix: number[][]
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
  xai:            XAIItem[]
  sensor_votes:   Record<string, SensorVote>
  fusion:         FusionResult
  weather_impact: Record<string, number>
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

export const WEATHER_OPTS = ['Clear', 'Cloudy', 'Rainy'] as const
export const THERMAL_OPTS = ['Not_Detected', 'Low', 'Medium', 'High'] as const
export const SIGNATURES   = ['IFF_MODE_5', 'IFF_MODE_3C', 'NO_IFF_RESPONSE', 'HOSTILE_JAMMING', 'UNKNOWN_EMISSION'] as const
export const PROFILES     = ['STABLE_CRUISE', 'AGGRESSIVE_MANEUVERS', 'LOW_ALTITUDE_FLYING', 'CLIMBING'] as const
