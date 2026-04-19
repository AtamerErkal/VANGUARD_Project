export interface SimPos {
  x: number
  y: number
}

export interface SimTrack {
  track_id:          string
  submitted_at:      string        // "HH:MM:SS UTC"
  ai_class:          string
  ai_conf:           number
  pos:               SimPos
  sensor_votes:      Record<string, { label: string; icon: string; vote: string; conf: number; reading: string }>
  // kinematic / ew / env fields echoed back
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
