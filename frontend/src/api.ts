import type { Track, PredictRequest, PredictResponse, ModelStats } from './types'

const BASE = import.meta.env.VITE_API_URL ?? ''

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`)
  return res.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`)
  return res.json()
}

export const api = {
  getTracks:     ()                    => get<Track[]>('/api/tracks'),
  getModelStats: ()                    => get<ModelStats>('/api/model-stats'),
  predict:       (req: PredictRequest) => post<PredictResponse>('/api/predict', req),
}
