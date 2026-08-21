import type { ArchitectureInput, DSSConfig } from './types'
import type { FieldErrors } from './components/ArchitectureForm'

const integerFields: Array<keyof ArchitectureInput> = ['num_planes', 'sats_per_plane', 'civilian_count', 'government_count', 'military_count']
const percentageFields: Array<keyof ArchitectureInput> = ['satellite_node_failure_probability', 'satellite_edge_failure_probability', 'ground_station_failure_probability', 'required_minimum_connectivity']

export function validateArchitecture(form: ArchitectureInput, config: DSSConfig): FieldErrors {
  const errors: FieldErrors = {}
  const checkRange = (field: keyof ArchitectureInput, range: { min: number; max: number } | undefined) => {
    const value = form[field]
    if (!Number.isFinite(value)) errors[field] = 'Enter a finite number.'
    else if (range && (value < range.min || value > range.max)) errors[field] = `Must be between ${range.min} and ${range.max}.`
  }
  for (const field of integerFields) {
    const range = config.space_domain[field] || undefined
    checkRange(field, range)
    if (!errors[field] && !Number.isInteger(form[field])) errors[field] = 'Enter a whole number.'
    if (!errors[field] && form[field] < 0) errors[field] = 'Must be nonnegative.'
  }
  for (const field of ['altitude_km', 'inclination_deg'] as const) checkRange(field, config.space_domain[field])
  for (const field of percentageFields) checkRange(field, config.space_domain[field] || config.ground_domain[field] || { min: 0, max: 1 })
  const capacities = config.ground_catalog_capacity
  for (const [field, key] of [['civilian_count', 'civilian'], ['government_count', 'government'], ['military_count', 'military']] as const) {
    if (!errors[field] && capacities?.[key] !== undefined && form[field] > capacities[key]) errors[field] = `Exceeds catalog capacity (${capacities[key]}).`
  }
  if (!errors.civilian_count && !errors.government_count && !errors.military_count && form.civilian_count + form.government_count + form.military_count < 1) errors.general = 'At least one ground station is required.'
  return errors
}

export const hasErrors = (errors: FieldErrors): boolean => Object.keys(errors).length > 0
