import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const config = {
  realization_count: 5,
  default_required_minimum_connectivity: 0.8,
  space_domain: {
    num_planes: { min: 4, max: 6 }, sats_per_plane: { min: 5, max: 8 }, altitude_km: { min: 300, max: 1200 }, inclination_deg: { min: 30, max: 98 }, satellite_node_failure_probability: { min: 0, max: 0.2 }, satellite_edge_failure_probability: { min: 0, max: 0.25 },
  },
  ground_domain: { ground_station_failure_probability: { min: 0, max: 0.4 } },
  ground_catalog_capacity: { civilian: 20, government: 20, military: 20 },
  model: { family: 'TGNN', target: 'space_gcc_fraction_original_min' },
}

const architecture = {
  num_planes: 5, sats_per_plane: 7, altitude_km: 550, inclination_deg: 53,
  satellite_node_failure_probability: 0.1, satellite_edge_failure_probability: 0.12,
  civilian_count: 10, government_count: 10, military_count: 10, ground_station_failure_probability: 0.08,
  required_minimum_connectivity: 0.8,
}

const result = {
  architecture,
  model: { family: 'TGNN', task: 'space_regression', target: 'space_gcc_fraction_original_min', checkpoint_sha256: 'abcdef1234567890', realization_count: 5 },
  space_resilience: { expected_minimum_gcc: 0.82, lowest_modeled_gcc: 0.76, highest_modeled_gcc: 0.88, required_minimum_connectivity: 0.8, expected_margin: 0.02, lowest_margin: -0.04, expected_assessment: 'MEETS_EXPECTED_REQUIREMENT', realizations_meeting_requirement: 4, realization_count: 5, realization_risk_flag: true },
  system_context: { mean_ground_service_fraction: 0.76, minimum_ground_service_fraction: 0.69, mean_overall_service_fraction: 0.74, minimum_overall_service_fraction: 0.68, limiting_segment: 'GROUND', ground_provenance: 'SATNET_CALCULATED', overall_provenance: 'SATNET_CALCULATED', status: 'AVAILABLE' },
  provenance: { prediction_provenance: 'TGNN_PREDICTION', training_performed: false },
  analysis_details: { realization_predictions: [0.84, 0.81, 0.76, 0.83, 0.84], realization_seeds: [1, 2, 3, 4, 5] },
}

const blockedResult = { ...result, system_context: { ...result.system_context, status: 'BLOCKED', blocked_reason: 'Ground service calculations were unavailable.' } }

function response(body: unknown, ok = true, status = 200) {
  return Promise.resolve({ ok, status, json: () => Promise.resolve(body) } as Response)
}

function mockApi(resultBody: unknown = result, ready = true, delay = 0) {
  const calls: Array<{ url: string; init?: RequestInit }> = []
  vi.stubGlobal('fetch', vi.fn((url: string, init?: RequestInit) => {
    calls.push({ url, init })
    if (url.endsWith('/config')) return response(config)
    if (url.endsWith('/readiness')) return response(ready ? { status: 'READY' } : { status: 'NOT_READY', reasons: ['Service inputs are unavailable.'] }, ready, ready ? 200 : 503)
    if (url.endsWith('/analyze')) return delay ? new Promise((resolve) => setTimeout(() => resolve(response(resultBody)), delay)) : response(resultBody)
    return response({}, false, 404)
  }))
  return calls
}

async function renderReady() {
  const calls = mockApi()
  render(<App />)
  await waitFor(() => expect(screen.getByTestId('analyze-button')).toBeEnabled())
  return calls
}

describe('SATNET DSS UI', () => {
  afterEach(() => { cleanup(); vi.restoreAllMocks() })

  it('loads the authoritative config and displays the default 80% threshold', async () => {
    await renderReady()
    expect(screen.getByTestId('field-required_minimum_connectivity')).toHaveValue(80)
    expect(screen.getByText(/Validated SATNET domain: 300 km–1200 km/)).toBeInTheDocument()
  })

  it('disables analysis when readiness is NOT_READY', async () => {
    mockApi(result, false)
    render(<App />)
    expect(await screen.findByText(/SATNET analysis service is not ready/)).toBeInTheDocument()
    expect(screen.getByTestId('analyze-button')).toBeDisabled()
  })

  it('converts percentage inputs to API fractions in the valid request payload', async () => {
    const calls = await renderReady()
    await userEvent.click(screen.getByTestId('analyze-button'))
    await waitFor(() => expect(calls.some((call) => call.url.endsWith('/analyze'))).toBe(true))
    const request = calls.find((call) => call.url.endsWith('/analyze'))!
    expect(JSON.parse(String(request.init?.body))).toMatchObject({ satellite_node_failure_probability: 0.1, satellite_edge_failure_probability: 0.12, ground_station_failure_probability: 0.08, required_minimum_connectivity: 0.8 })
  })

  it('shows a loading state and prevents duplicate analysis', async () => {
    const calls = mockApi(result, true, 100)
    render(<App />)
    await waitFor(() => expect(screen.getByTestId('analyze-button')).toBeEnabled())
    const button = screen.getByTestId('analyze-button')
    await userEvent.click(button)
    expect(button).toBeDisabled()
    expect(screen.getByText('Evaluating architecture...')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByTestId('expected-minimum-gcc')).toBeInTheDocument())
    expect(calls.filter((call) => call.url.endsWith('/analyze'))).toHaveLength(1)
  })

  it('renders expected connectivity, threshold, margin, assessment, and realization warning', async () => {
    await renderReady()
    await userEvent.click(screen.getByTestId('analyze-button'))
    expect(await screen.findByTestId('expected-minimum-gcc')).toHaveTextContent('82%')
    expect(screen.getByText('Required Connectivity')).toBeInTheDocument()
    expect(screen.getByText('+2 percentage points')).toBeInTheDocument()
    expect(screen.getByText('MEETS EXPECTED REQUIREMENT')).toBeInTheDocument()
    expect(screen.getByText('1 of 5 modeled failure realizations fell below the requirement.')).toBeInTheDocument()
    expect(screen.queryByText(/probability|likelihood/i)).not.toBeInTheDocument()
  })

  it('renders available system context and keeps details collapsed by default', async () => {
    await renderReady()
    await userEvent.click(screen.getByTestId('analyze-button'))
    await screen.findByTestId('expected-minimum-gcc')
    expect(screen.getByText('Mean Ground Service')).toBeInTheDocument()
    expect(screen.getByText('76%')).toBeInTheDocument()
    expect(screen.getByText('GROUND')).toBeInTheDocument()
    expect(screen.getByText('SATNET-calculated ground/system metrics')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Analysis Details/ })).toBeInTheDocument()
    expect(screen.queryByText(/Raw modeled GCC predictions/)).not.toBeVisible()
  })

  it('renders a blocked system context without inventing values', async () => {
    mockApi(blockedResult)
    render(<App />)
    await waitFor(() => expect(screen.getByTestId('analyze-button')).toBeEnabled())
    await userEvent.click(screen.getByTestId('analyze-button'))
    expect(await screen.findByText('Ground/system context unavailable for this analysis. Ground service calculations were unavailable.')).toBeInTheDocument()
    expect(screen.queryByText('Mean Ground Service')).not.toBeInTheDocument()
  })

  it('saves and removes a comparison while preserving the design threshold', async () => {
    await renderReady()
    await userEvent.click(screen.getByTestId('analyze-button'))
    await screen.findByTestId('expected-minimum-gcc')
    const label = screen.getByLabelText('Comparison design label')
    await userEvent.clear(label)
    await userEvent.type(label, 'High-Resilience Option')
    await userEvent.click(screen.getByRole('button', { name: /Save to Comparison/ }))
    expect(screen.getByText('High-Resilience Option')).toBeInTheDocument()
    expect(screen.getByText(/saved designs/)).toHaveTextContent('1/4')
    const table = screen.getByRole('table')
    expect(within(table).getByText('80%')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: /Remove High-Resilience Option/ }))
    expect(screen.getByText(/saved designs/)).toHaveTextContent('0/4')
  })

  it('limits client-side comparison storage to four designs', async () => {
    await renderReady()
    await userEvent.click(screen.getByTestId('analyze-button'))
    await screen.findByTestId('expected-minimum-gcc')
    const save = screen.getByRole('button', { name: /Save to Comparison/ })
    await userEvent.click(save)
    await userEvent.click(save)
    await userEvent.click(save)
    await userEvent.click(save)
    expect(screen.getByText(/saved designs/)).toHaveTextContent('4/4')
    await userEvent.click(save)
    expect(screen.getByText(/saved designs/)).toHaveTextContent('4/4')
  })

  it('shows API errors without exposing internals', async () => {
    const calls = mockApi()
    vi.mocked(fetch).mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      calls.push({ url, init })
      if (url.endsWith('/config')) return response(config)
      if (url.endsWith('/readiness')) return response({ status: 'READY' })
      return response({ error: { code: 'ARCHITECTURE_OUTSIDE_VALIDATED_DOMAIN', message: 'Altitude is outside the validated domain.' } }, false, 422)
    })
    render(<App />)
    await waitFor(() => expect(screen.getByTestId('analyze-button')).toBeEnabled())
    await userEvent.click(screen.getByTestId('analyze-button'))
    expect((await screen.findAllByText('Altitude is outside the validated domain.')).length).toBeGreaterThan(0)
    expect(screen.queryByText(/traceback|checkpoint path/i)).not.toBeInTheDocument()
  })

  it('shows inline range validation before submitting', async () => {
    mockApi()
    render(<App />)
    await waitFor(() => expect(screen.getByTestId('analyze-button')).toBeEnabled())
    fireEvent.change(screen.getByTestId('field-altitude_km'), { target: { value: '1300' } })
    await userEvent.click(screen.getByTestId('analyze-button'))
    expect(await screen.findByText('Must be between 300 and 1200.')).toBeInTheDocument()
    expect(screen.queryByText('Expected Minimum Connectivity')).not.toBeInTheDocument()
  })
})
