import type { AnalysisResult, ArchitectureInput, ApiErrorPayload, DSSConfig, Readiness } from './types'

export const API_BASE_URL = (import.meta.env.VITE_SATNET_API_BASE_URL || 'http://localhost:8000/api/v1').replace(/\/$/, '')
const REQUEST_TIMEOUT_MS = 30_000

export class ApiError extends Error {
  readonly status: number
  readonly code?: string
  readonly field?: string

  constructor(message: string, status: number, payload?: ApiErrorPayload) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = payload?.error?.code
    this.field = payload?.error?.field
  }
}

async function fetchWithTimeout(url: string, options?: RequestInit): Promise<Response> {
  const controller = new AbortController()
  const timeout = window.setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS)
  try {
    return await fetch(url, { ...options, signal: controller.signal })
  } finally {
    window.clearTimeout(timeout)
  }
}

async function requestJson<T>(path: string, options?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetchWithTimeout(`${API_BASE_URL}${path}`, {
      ...options,
      headers: { 'Content-Type': 'application/json', ...options?.headers },
    })
  } catch {
    throw new ApiError('The SATNET analysis service could not be reached or timed out.', 0)
  }

  let body: T | ApiErrorPayload | undefined
  try {
    body = await response.json()
  } catch {
    body = undefined
  }
  if (!response.ok) {
    const payload = body as ApiErrorPayload | undefined
    throw new ApiError(
      payload?.error?.message || `The SATNET service returned an error (${response.status}).`,
      response.status,
      payload,
    )
  }
  return body as T
}

export function getConfig(): Promise<DSSConfig> {
  return requestJson<DSSConfig>('/config')
}

export async function getReadiness(): Promise<Readiness> {
  let response: Response
  try {
    response = await fetchWithTimeout(`${API_BASE_URL}/readiness`, { headers: { 'Content-Type': 'application/json' } })
  } catch {
    throw new ApiError('The SATNET analysis service could not be reached or timed out.', 0)
  }
  let body: Readiness
  try {
    body = (await response.json()) as Readiness
  } catch {
    throw new ApiError('The SATNET service returned an unreadable readiness response.', response.status)
  }
  if (!response.ok && response.status !== 503) {
    throw new ApiError('The SATNET service returned an unexpected readiness response.', response.status)
  }
  return body
}

export function analyzeArchitecture(input: ArchitectureInput): Promise<AnalysisResult> {
  return requestJson<AnalysisResult>('/analyze', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}

export async function loadStartup(): Promise<{ config: DSSConfig; readiness: Readiness }> {
  const [config, readiness] = await Promise.all([getConfig(), getReadiness()])
  return { config, readiness }
}
