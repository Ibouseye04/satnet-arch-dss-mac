import {
  Box,
  Card,
  CardContent,
  Divider,
  Grid,
  InputAdornment,
  Stack,
  TextField,
  Typography,
} from '@mui/material'
import type { DSSConfig, ArchitectureInput } from '../types'

export type FieldErrors = Partial<Record<keyof ArchitectureInput | 'general', string>>

type Props = {
  form: ArchitectureInput
  config: DSSConfig
  errors: FieldErrors
  onChange: (field: keyof ArchitectureInput, value: number) => void
}

const percentFields: Array<{ field: keyof ArchitectureInput; label: string; helper: string }> = [
  {
    field: 'satellite_node_failure_probability',
    label: 'Satellite Node Failure',
    helper: 'Validated SATNET domain: 0–20%',
  },
  {
    field: 'satellite_edge_failure_probability',
    label: 'Satellite Edge Failure',
    helper: 'Maximum modeled edge-failure rate: 25%',
  },
]

const rangeText = (config: DSSConfig, field: string, unit: string): string => {
  const range = config.space_domain[field] || config.ground_domain[field]
  return range ? `Validated SATNET domain: ${range.min}${unit}–${range.max}${unit}` : ''
}

function NumberField({
  field,
  label,
  value,
  error,
  helperText,
  min,
  max,
  step = 1,
  unit,
  onChange,
}: {
  field: keyof ArchitectureInput
  label: string
  value: number
  error?: string
  helperText?: string
  min?: number
  max?: number
  step?: number
  unit?: string
  onChange: Props['onChange']
}) {
  return (
    <TextField
      fullWidth
      type="number"
      label={label}
      value={value}
      error={Boolean(error)}
      helperText={error || helperText}
      inputProps={{ min, max, step, 'data-testid': `field-${field}` }}
      InputProps={unit ? { endAdornment: <InputAdornment position="end">{unit}</InputAdornment> } : undefined}
      onChange={(event) => onChange(field, Number(event.target.value))}
    />
  )
}

export default function ArchitectureForm({ form, config, errors, onChange }: Props) {
  const space = config.space_domain
  const ground = config.ground_domain
  const capacity = config.ground_catalog_capacity
  const countProps = (field: keyof ArchitectureInput, label: string, key: string) => ({
    field,
    label,
    value: form[field] as number,
    error: errors[field],
    helperText: capacity?.[key] !== undefined ? `Catalog capacity: ${capacity[key]}` : 'Nonnegative integer',
    min: 0,
    max: capacity?.[key],
    step: 1,
    onChange,
  })

  return (
    <Card component="section" aria-labelledby="configuration-title" className="panel-card">
      <CardContent>
        <Typography id="configuration-title" variant="h5" component="h2" className="section-title">
          Architecture Configuration
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2.5 }}>
          Define a candidate space and ground architecture within the qualified SATNET domain.
        </Typography>

        <Typography variant="overline" className="eyebrow">Space segment</Typography>
        <Grid container spacing={2} sx={{ mb: 2.5 }}>
          <Grid size={{ xs: 12, sm: 6 }}>
            <NumberField field="num_planes" label="Orbital Planes" value={form.num_planes} error={errors.num_planes} helperText="Integer validated by API configuration" min={space.num_planes?.min} max={space.num_planes?.max} onChange={onChange} />
          </Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <NumberField field="sats_per_plane" label="Satellites per Plane" value={form.sats_per_plane} error={errors.sats_per_plane} helperText="Integer validated by API configuration" min={space.sats_per_plane?.min} max={space.sats_per_plane?.max} onChange={onChange} />
          </Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <NumberField field="altitude_km" label="Altitude" value={form.altitude_km} error={errors.altitude_km} helperText={rangeText(config, 'altitude_km', ' km')} min={space.altitude_km?.min} max={space.altitude_km?.max} step={1} unit="km" onChange={onChange} />
          </Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <NumberField field="inclination_deg" label="Inclination" value={form.inclination_deg} error={errors.inclination_deg} helperText={rangeText(config, 'inclination_deg', '°')} min={space.inclination_deg?.min} max={space.inclination_deg?.max} step={0.1} unit="°" onChange={onChange} />
          </Grid>
          {percentFields.map(({ field, label, helper }) => {
            const bounds = space[field]
            return (
              <Grid size={{ xs: 12, sm: 6 }} key={field}>
                <NumberField field={field} label={label} value={Math.round((form[field] as number) * 10000) / 100} error={errors[field]} helperText={errors[field] || helper} min={(bounds?.min ?? 0) * 100} max={(bounds?.max ?? 1) * 100} step={1} unit="%" onChange={(name, value) => onChange(name, value / 100)} />
              </Grid>
            )
          })}
        </Grid>

        <Divider sx={{ mb: 2.2 }} />
        <Typography variant="overline" className="eyebrow">Ground segment</Typography>
        <Grid container spacing={2} sx={{ mb: 2.5 }}>
          <Grid size={{ xs: 12, sm: 4 }}><NumberField {...countProps('civilian_count', 'Civilian Ground Stations', 'civilian')} /></Grid>
          <Grid size={{ xs: 12, sm: 4 }}><NumberField {...countProps('government_count', 'Government Ground Stations', 'government')} /></Grid>
          <Grid size={{ xs: 12, sm: 4 }}><NumberField {...countProps('military_count', 'Military Ground Stations', 'military')} /></Grid>
          <Grid size={{ xs: 12, sm: 6 }}>
            <NumberField field="ground_station_failure_probability" label="Ground Station Failure" value={Math.round(form.ground_station_failure_probability * 10000) / 100} error={errors.ground_station_failure_probability} helperText={rangeText(config, 'ground_station_failure_probability', '%')} min={(ground.ground_station_failure_probability?.min ?? 0) * 100} max={(ground.ground_station_failure_probability?.max ?? 1) * 100} step={1} unit="%" onChange={(name, value) => onChange(name, value / 100)} />
          </Grid>
        </Grid>

        <Divider sx={{ mb: 2.2 }} />
        <Typography variant="overline" className="eyebrow">Mission requirement</Typography>
        <Box sx={{ maxWidth: 380 }}>
          <NumberField field="required_minimum_connectivity" label="Required Minimum Connectivity" value={Math.round(form.required_minimum_connectivity * 10000) / 100} error={errors.required_minimum_connectivity} helperText={errors.required_minimum_connectivity || 'Default reflects the SATNET experimental resilience threshold. Adjust to match mission requirements.'} min={0} max={100} step={1} unit="%" onChange={(name, value) => onChange(name, value / 100)} />
        </Box>
        {errors.general && <Typography color="error" variant="body2" sx={{ mt: 2 }}>{errors.general}</Typography>}
      </CardContent>
    </Card>
  )
}
