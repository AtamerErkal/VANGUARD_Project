import type { XAIItem, SensorVote, FusionResult, Anomaly } from './types'

export interface SimPos {
  x: number
  y: number
}

export interface SimTrack {
  track_id:          string
  submitted_at:      string        // "HH:MM:SS UTC"
  ai_class:          string
  ai_conf:           number
  ai_probs:          Record<string, number>
  pos:               SimPos
  sensor_votes:      Record<string, SensorVote>
  fusion:            FusionResult
  xai:               XAIItem[]
  // kinematic / ew / env
  altitude_ft:       number
  speed_kts:         number
  rcs_m2:            number
  heading:           number
  esm_signature:     string
  iff_mode:          string
  flight_profile:    string
  weather:           string
  thermal_signature: string
  // anomaly detection (backend-submitted tracks only)
  anomalies?:        Anomaly[]
  // trajectory history (backend-submitted; demo tracks use synthetic fallback)
  hist_alts?:        number[]
  hist_speeds?:      number[]
  hist_headings?:    number[]
  hist_timestamps?:  string[]
}

export interface SimSubmitRequest {
  altitude_ft:       number
  speed_kts:         number
  rcs_m2:            number
  heading:           number
  esm_signature:     string
  iff_mode:          string
  flight_profile:    string
  weather:           string
  thermal_signature: string
}
