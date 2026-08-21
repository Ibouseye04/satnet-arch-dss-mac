import {
  Accordion,
  AccordionDetails,
  AccordionSummary,
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  Divider,
  Grid,
  Stack,
  Typography,
} from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import SaveAltIcon from '@mui/icons-material/SaveAlt'
import type { AnalysisResult, ArchitectureInput } from '../types'
import { abbreviatedHash, formatMargin, formatPercent, safeBlockedReason } from '../format'

function ResilienceBar({ prediction, threshold }: { prediction: number; threshold: number }) {
  const predictionPosition = Math.max(0, Math.min(100, prediction * 100))
  const thresholdPosition = Math.max(0, Math.min(100, threshold * 100))
  return (
    <Box className="resilience-visual" aria-label={`Expected connectivity ${formatPercent(prediction)}, requirement ${formatPercent(threshold)}`}>
      <Box className="resilience-scale"><span>0%</span><span>100%</span></Box>
      <Box className="resilience-track">
        <Box className="resilience-fill" style={{ width: `${predictionPosition}%` }} />
        <Box className="resilience-threshold" style={{ left: `${thresholdPosition}%` }} aria-hidden="true" />
        <Box className="resilience-prediction" style={{ left: `${predictionPosition}%` }} aria-hidden="true" />
      </Box>
      <Box className="resilience-labels">
        <span style={{ left: `${thresholdPosition}%` }}>Requirement {formatPercent(threshold)}</span>
        <span style={{ left: `${predictionPosition}%` }}>Prediction {formatPercent(prediction)}</span>
      </Box>
    </Box>
  )
}

function Metric({ label, value, emphasis = false }: { label: string; value: string; emphasis?: boolean }) {
  return <Box className={emphasis ? 'metric metric-emphasis' : 'metric'}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant={emphasis ? 'h3' : 'h6'}>{value}</Typography></Box>
}

function SystemContext({ result }: { result: AnalysisResult }) {
  const context = result.system_context
  if (context.status === 'BLOCKED') {
    const reason = safeBlockedReason(context.blocked_reason)
    return (
      <Card component="section" aria-labelledby="system-context-title" className="panel-card">
        <CardContent>
          <Typography id="system-context-title" variant="h6" component="h2" className="section-title">System Context</Typography>
          <Alert severity="warning" sx={{ mt: 2 }}>Ground/system context unavailable for this analysis.{reason ? ` ${reason}` : ''}</Alert>
        </CardContent>
      </Card>
    )
  }
  return (
    <Card component="section" aria-labelledby="system-context-title" className="panel-card">
      <CardContent>
        <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'flex-start', sm: 'center' }} gap={1}>
          <Typography id="system-context-title" variant="h6" component="h2" className="section-title">System Context</Typography>
          <Chip size="small" variant="outlined" label="SATNET-calculated ground/system metrics" />
        </Stack>
        <Grid container spacing={2.5} sx={{ mt: 1 }}>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}><Metric label="Mean Ground Service" value={formatPercent(context.mean_ground_service_fraction)} /></Grid>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}><Metric label="Minimum Ground Service" value={formatPercent(context.minimum_ground_service_fraction)} /></Grid>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}><Metric label="Mean Overall Service" value={formatPercent(context.mean_overall_service_fraction)} /></Grid>
          <Grid size={{ xs: 12, sm: 6, md: 3 }}><Metric label="Minimum Overall Service" value={formatPercent(context.minimum_overall_service_fraction)} /></Grid>
        </Grid>
        <Divider sx={{ my: 2 }} />
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={3}>
          <Box><Typography variant="caption" color="text.secondary">Limiting Segment</Typography><Typography fontWeight={700}>{context.limiting_segment || '—'}</Typography></Box>
          <Box><Typography variant="caption" color="text.secondary">Space resilience provenance</Typography><Typography fontWeight={600}>TGNN Prediction</Typography></Box>
          <Box><Typography variant="caption" color="text.secondary">Ground/system provenance</Typography><Typography fontWeight={600}>SATNET Calculated</Typography></Box>
        </Stack>
      </CardContent>
    </Card>
  )
}

function ArchitectureSummary({ architecture }: { architecture: ArchitectureInput }) {
  return (
    <Card variant="outlined" className="summary-card">
      <CardContent>
        <Typography variant="overline" className="eyebrow">Evaluated architecture</Typography>
        <Typography variant="body1" fontWeight={700}>{architecture.num_planes} planes × {architecture.sats_per_plane} satellites · {architecture.num_planes * architecture.sats_per_plane} total satellites</Typography>
        <Typography variant="body2" color="text.secondary">{architecture.altitude_km} km · {architecture.inclination_deg}° inclination</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Node failure: {formatPercent(architecture.satellite_node_failure_probability)} · Edge failure: {formatPercent(architecture.satellite_edge_failure_probability)}</Typography>
        <Typography variant="body2" color="text.secondary">Ground: {architecture.civilian_count} civilian · {architecture.government_count} government · {architecture.military_count} military · {formatPercent(architecture.ground_station_failure_probability)} failure</Typography>
      </CardContent>
    </Card>
  )
}

function AnalysisDetails({ result }: { result: AnalysisResult }) {
  const details = result.analysis_details
  const predictions = Array.isArray(details.realization_predictions) ? details.realization_predictions as number[] : []
  const seeds = Array.isArray(details.realization_seeds) ? details.realization_seeds : []
  return (
    <Accordion className="details-accordion">
      <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls="analysis-details-content" id="analysis-details-header">Analysis Details</AccordionSummary>
      <AccordionDetails id="analysis-details-content">
        <Grid container spacing={2}>
          <Grid size={{ xs: 12, md: 6 }}>
            <Typography variant="subtitle2">Raw modeled GCC predictions</Typography>
            <Typography variant="body2" color="text.secondary">{predictions.map((value, index) => `R${index + 1}: ${formatPercent(value)}`).join(' · ') || 'Not supplied by API'}</Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>Lowest: {formatPercent(result.space_resilience.lowest_modeled_gcc)} · Highest: {formatPercent(result.space_resilience.highest_modeled_gcc)}</Typography>
            {seeds.length > 0 && <Typography variant="body2" color="text.secondary">Seeds: {seeds.join(', ')}</Typography>}
          </Grid>
          <Grid size={{ xs: 12, md: 6 }}>
            <Typography variant="subtitle2">Model and provenance</Typography>
            <Typography variant="body2" color="text.secondary">Family: {result.model.family} · Target: {result.model.target}</Typography>
            <Typography variant="body2" color="text.secondary">Realizations: {result.model.realization_count} · Checkpoint: {abbreviatedHash(result.model.checkpoint_sha256)}</Typography>
            <Typography variant="body2" color="text.secondary">Space: TGNN Prediction · Ground/system: SATNET Calculated</Typography>
          </Grid>
        </Grid>
      </AccordionDetails>
    </Accordion>
  )
}

export default function ResultCard({ result, architecture, comparisonLabel, onLabelChange, onSave }: { result: AnalysisResult; architecture: ArchitectureInput; comparisonLabel: string; onLabelChange: (label: string) => void; onSave: () => void }) {
  const space = result.space_resilience
  const meets = space.expected_assessment === 'MEETS_EXPECTED_REQUIREMENT'
  const failedRealizations = space.realization_count - space.realizations_meeting_requirement
  return (
    <Stack spacing={2}>
      <Card component="section" className={`result-card ${meets ? 'result-meets' : 'result-below'}`} aria-labelledby="resilience-title">
        <CardContent>
          <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'flex-start', sm: 'center' }} gap={2}>
            <Box>
              <Typography variant="overline" className="eyebrow">Resilience assessment</Typography>
              <Typography id="resilience-title" variant="h6" component="h2">Expected Minimum Connectivity</Typography>
            </Box>
            <Chip color={meets ? 'success' : 'error'} label={meets ? 'MEETS EXPECTED REQUIREMENT' : 'BELOW EXPECTED REQUIREMENT'} />
          </Stack>
          <Box className="hero-metric"><Typography data-testid="expected-minimum-gcc" variant="h1">{formatPercent(space.expected_minimum_gcc)}</Typography><Typography variant="body2" color="text.secondary">Mean of five TGNN-predicted minimum GCC values</Typography></Box>
          <ResilienceBar prediction={space.expected_minimum_gcc} threshold={space.required_minimum_connectivity} />
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid size={{ xs: 12, sm: 4 }}><Metric label="Required Connectivity" value={formatPercent(space.required_minimum_connectivity)} /></Grid>
            <Grid size={{ xs: 12, sm: 4 }}><Metric label="Expected Margin" value={formatMargin(space.expected_margin)} emphasis /></Grid>
            <Grid size={{ xs: 12, sm: 4 }}><Metric label="Assessment" value={meets ? 'MEETS' : 'BELOW'} /></Grid>
          </Grid>
          {failedRealizations > 0 ? <Alert severity="warning" sx={{ mt: 2 }}>{failedRealizations} of {space.realization_count} modeled failure realizations fell below the requirement.</Alert> : <Alert severity="success" sx={{ mt: 2 }}>All {space.realization_count} modeled failure realizations met the requirement.</Alert>}
        </CardContent>
      </Card>
      <ArchitectureSummary architecture={architecture} />
      <Card variant="outlined" className="save-card">
        <CardContent>
          <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ xs: 'stretch', sm: 'center' }} spacing={1.5}>
            <Box sx={{ flex: 1 }}><Typography variant="subtitle2">Keep this result for architecture comparison</Typography><Typography variant="caption" color="text.secondary">Comparison is stored in this browser session only.</Typography></Box>
            <input className="comparison-label" aria-label="Comparison design label" value={comparisonLabel} onChange={(event) => onLabelChange(event.target.value)} />
            <Button variant="outlined" startIcon={<SaveAltIcon />} onClick={onSave}>Save to Comparison</Button>
          </Stack>
        </CardContent>
      </Card>
      <SystemContext result={result} />
      <AnalysisDetails result={result} />
    </Stack>
  )
}
