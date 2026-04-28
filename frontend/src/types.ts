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

export interface TrackQuality {
  label:             'HIGH' | 'MEDIUM' | 'LOW'
  uncertainty:       number
  covariance_trace:  number
  smoothed_position: { lat: number; lon: number; alt_ft: number }
  estimated_velocity: { dlat_per_step: number; dlon_per_step: number }
}

export interface DSFusion {
  best:          string
  probs:         Record<string, number>
  conflict_mass: number
}

export interface Track {
  track_id:             string
  latitude:             number
  longitude:            number
  altitude_ft:          number
  speed_kts:            number
  rcs_m2:               number
  heading:              number
  weather:          string
  esm_signature:    string
  iff_mode:         string
  flight_profile:   string
  thermal_signature:    string
  ai_class:             string
  ai_conf:              number
  ai_probs:             Record<string, number>
  epistemic_uncertainty?: number
  uncertainty_label?:     'LOW' | 'MEDIUM' | 'HIGH'
  sensor_votes:         Record<string, SensorVote>
  fusion:               FusionResult
  ds_fusion?:           DSFusion
  track_quality?:       TrackQuality
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
  altitude_ft:       number
  speed_kts:         number
  rcs_m2:            number
  latitude:          number
  longitude:         number
  heading:           number
  esm_signature:     string
  iff_mode:          string
  flight_profile:    string
  weather:           string
  thermal_signature: string
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

// NATO standard air picture classification (STANAG APP-6)
export const CLASS_STYLES: Record<string, { color: string; bg: string; icon: string; nato: string }> = {
  'HOSTILE':        { color: '#f87171', bg: '#3f0808', icon: '◆', nato: 'H' },
  'SUSPECT':        { color: '#fb923c', bg: '#3b1003', icon: '◈', nato: 'S' },
  'UNKNOWN':        { color: '#c4b5fd', bg: '#2d1b69', icon: '◻', nato: 'U' },
  'NEUTRAL':        { color: '#94a3b8', bg: '#1e293b', icon: '▬', nato: 'N' },
  'ASSUMED FRIEND': { color: '#34d399', bg: '#053f2e', icon: '◉', nato: 'A' },
  'FRIEND':         { color: '#4ade80', bg: '#0f3320', icon: '●', nato: 'F' },
}

export const SENSOR_ORDER = ['radar', 'esm', 'irst', 'iff'] as const

export const WEATHER_OPTS = ['Clear', 'Cloudy', 'Rainy'] as const
export const THERMAL_OPTS = ['Not_Detected', 'Low', 'Medium', 'High'] as const
export const ESM_SIGS     = ['CLEAN', 'UNKNOWN_EMISSION', 'NOISE_JAMMING', 'HOSTILE_JAMMING'] as const
export const IFF_MODES    = ['IFF_MODE_5', 'IFF_MODE_3C', 'DEGRADED', 'NO_RESPONSE'] as const
export const PROFILES     = ['STABLE_CRUISE', 'AGGRESSIVE_MANEUVERS', 'LOW_ALTITUDE_FLYING', 'CLIMBING'] as const
