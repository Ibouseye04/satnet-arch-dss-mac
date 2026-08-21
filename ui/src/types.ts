export type DomainRange = { min: number; max: number }

export type DSSConfig = {
  realization_count: number
  default_required_minimum_connectivity: number
  space_domain: Record<string, DomainRange>
  ground_domain: Record<string, DomainRange>
  ground_catalog_capacity?: Record<string, number>
  model: { family: string; target: string }
}

export type Readiness = {
  status: 'READY' | 'NOT_READY'
  service?: string
  api_version?: string
  reasons?: string[]
}

export type ArchitectureInput = {
  num_planes: number
  sats_per_plane: number
  altitude_km: number
  inclination_deg: number
  satellite_node_failure_probability: number
  satellite_edge_failure_probability: number
  civilian_count: number
  government_count: number
  military_count: number
  ground_station_failure_probability: number
  required_minimum_connectivity: number
}

export type SpaceResilience = {
  expected_minimum_gcc: number
  lowest_modeled_gcc: number
  highest_modeled_gcc: number
  required_minimum_connectivity: number
  expected_margin: number
  lowest_margin: number
  expected_assessment: 'MEETS_EXPECTED_REQUIREMENT' | 'BELOW_EXPECTED_REQUIREMENT' | string
  realizations_meeting_requirement: number
  realization_count: number
  realization_risk_flag: boolean
}

export type SystemContext = {
  mean_ground_service_fraction: number | null
  minimum_ground_service_fraction: number | null
  mean_overall_service_fraction: number | null
  minimum_overall_service_fraction: number | null
  limiting_segment: string | null
  ground_provenance: string
  overall_provenance: string
  status: 'AVAILABLE' | 'BLOCKED' | string
  blocked_reason?: string | null
}

export type AnalysisResult = {
  architecture: ArchitectureInput
  model: {
    family: string
    task: string
    target: string
    checkpoint_sha256: string
    realization_count: number
  }
  space_resilience: SpaceResilience
  system_context: SystemContext
  provenance: Record<string, unknown>
  analysis_details: Record<string, unknown>
}

export type SavedComparison = {
  id: string
  label: string
  architecture: ArchitectureInput
  result: AnalysisResult
}

export type ApiErrorPayload = {
  error?: { code?: string; message?: string; field?: string }
}
