import { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  AppBar,
  Box,
  Button,
  CircularProgress,
  Container,
  CssBaseline,
  Divider,
  Paper,
  Stack,
  Toolbar,
  Tooltip,
  Typography,
} from '@mui/material'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import ArchitectureIcon from '@mui/icons-material/AccountTreeOutlined'
import { createTheme, ThemeProvider } from '@mui/material/styles'
import ArchitectureForm from './components/ArchitectureForm'
import type { FieldErrors } from './components/ArchitectureForm'
import ComparisonPanel from './components/ComparisonPanel'
import MethodologyDialog from './components/MethodologyDialog'
import ResultCard from './components/ResultCard'
import { analyzeArchitecture, ApiError, getConfig, getReadiness } from './api'
import type { AnalysisResult, ArchitectureInput, DSSConfig, Readiness, SavedComparison } from './types'
import { hasErrors, validateArchitecture } from './validation'

const initialArchitecture: ArchitectureInput = {
  num_planes: 5,
  sats_per_plane: 7,
  altitude_km: 550,
  inclination_deg: 53,
  satellite_node_failure_probability: 0.10,
  satellite_edge_failure_probability: 0.12,
  civilian_count: 10,
  government_count: 10,
  military_count: 10,
  ground_station_failure_probability: 0.08,
  required_minimum_connectivity: 0.80,
}

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: '#0b466b', contrastText: '#fff' },
    secondary: { main: '#5f6b76' },
    success: { main: '#2f6b4f' },
    warning: { main: '#9a6500' },
    error: { main: '#a13d34' },
    background: { default: '#f3f6f8', paper: '#ffffff' },
  },
  typography: {
    fontFamily: 'Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
    h1: { fontWeight: 750, letterSpacing: '-0.04em' },
    h2: { fontWeight: 700 },
    h5: { fontWeight: 700 },
    h6: { fontWeight: 700 },
  },
  shape: { borderRadius: 8 },
  components: {
    MuiButton: { defaultProps: { disableElevation: true } },
    MuiCard: { styleOverrides: { root: { border: '1px solid #dce3e8', boxShadow: '0 2px 10px rgba(20, 45, 65, 0.04)' } } },
  },
})

function startupErrorMessage(error: unknown): string {
  return error instanceof ApiError ? error.message : 'The SATNET analysis service could not be reached.'
}

export default function App() {
  const [config, setConfig] = useState<DSSConfig | null>(null)
  const [readiness, setReadiness] = useState<Readiness | null>(null)
  const [startupError, setStartupError] = useState<string | null>(null)
  const [form, setForm] = useState<ArchitectureInput>(initialArchitecture)
  const [errors, setErrors] = useState<FieldErrors>({})
  const [result, setResult] = useState<AnalysisResult | null>(null)
  const [analysisError, setAnalysisError] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [methodologyOpen, setMethodologyOpen] = useState(false)
  const [comparisonLabel, setComparisonLabel] = useState('Design A')
  const [comparisons, setComparisons] = useState<SavedComparison[]>([])

  useEffect(() => {
    let active = true
    Promise.allSettled([getConfig(), getReadiness()]).then(([configOutcome, readinessOutcome]) => {
      if (!active) return
      if (configOutcome.status === 'fulfilled') setConfig(configOutcome.value)
      else setStartupError(startupErrorMessage(configOutcome.reason))
      if (readinessOutcome.status === 'fulfilled') setReadiness(readinessOutcome.value)
      else if (configOutcome.status === 'fulfilled') setStartupError(startupErrorMessage(readinessOutcome.reason))
    })
    return () => { active = false }
  }, [])

  useEffect(() => {
    if (config && !result) setForm((current) => ({ ...current, required_minimum_connectivity: config.default_required_minimum_connectivity }))
  }, [config, result])

  const isReady = readiness?.status === 'READY'
  const validationErrors = useMemo(() => config ? validateArchitecture(form, config) : {}, [form, config])

  const updateField = (field: keyof ArchitectureInput, value: number) => {
    setForm((current) => ({ ...current, [field]: value }))
    setErrors((current) => ({ ...current, [field]: undefined, general: undefined }))
  }

  const handleAnalyze = async () => {
    if (!config || !isReady) return
    const nextErrors = validateArchitecture(form, config)
    setErrors(nextErrors)
    if (hasErrors(nextErrors)) return
    setAnalysisError(null)
    setIsAnalyzing(true)
    const startedAt = performance.now()
    try {
      const analyzed = await analyzeArchitecture(form)
      setResult(analyzed)
      setComparisonLabel(`Design ${String.fromCharCode(65 + comparisons.length)}`)
      if (import.meta.env.DEV) console.info(`SATNET analysis completed in ${Math.round(performance.now() - startedAt)} ms`)
    } catch (error) {
      setAnalysisError(startupErrorMessage(error))
    } finally {
      setIsAnalyzing(false)
    }
  }

  const saveComparison = () => {
    if (!result || comparisons.length >= 4) return
    const label = comparisonLabel.trim() || `Design ${String.fromCharCode(65 + comparisons.length)}`
    setComparisons((current) => [...current, { id: crypto.randomUUID(), label, architecture: { ...result.architecture }, result }])
  }

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AppBar position="static" color="primary" className="top-bar">
        <Toolbar sx={{ minHeight: { xs: 64, sm: 74 } }}>
          <ArchitectureIcon sx={{ mr: 1.5, fontSize: 30 }} />
          <Box sx={{ flex: 1 }}>
            <Typography variant="h5" component="div" letterSpacing="0.06em">SATNET</Typography>
            <Typography variant="caption" sx={{ opacity: 0.84 }}>Satellite Network Architecture Decision Support System</Typography>
          </Box>
          <Tooltip title="Read methodology, assumptions, and model limitations">
            <Button color="inherit" startIcon={<InfoOutlinedIcon />} onClick={() => setMethodologyOpen(true)}>Methodology</Button>
          </Tooltip>
        </Toolbar>
      </AppBar>

      <Container maxWidth="xl" component="main" sx={{ py: { xs: 2, md: 4 } }}>
        <Stack spacing={2.5}>
          {startupError && <Alert severity="error" role="alert">{startupError}</Alert>}
          {readiness?.status === 'NOT_READY' && <Alert severity="warning" role="alert"><strong>SATNET analysis service is not ready.</strong> Analysis is disabled until the service reports READY. {readiness.reasons?.join(' ')}</Alert>}
          {isAnalyzing && <Paper className="progress-banner" role="status"><CircularProgress size={22} /><Box><Typography fontWeight={700}>Evaluating architecture...</Typography><Typography variant="caption" color="text.secondary">Running {config?.realization_count || 5} deterministic SATNET failure realizations and resilience inference.</Typography></Box></Paper>}

          <Box className="intro-row">
            <Box><Typography variant="h4" component="h1">Architecture resilience review</Typography><Typography color="text.secondary" sx={{ mt: 0.5 }}>Assess a candidate design against an explicit connectivity requirement.</Typography></Box>
            {config && <Typography variant="caption" color="text.secondary" className="api-status">API contract loaded · {config.realization_count} realizations</Typography>}
          </Box>

          <Box className="workspace-grid">
            <ArchitectureForm form={form} config={config || fallbackConfig} errors={errors} onChange={updateField} />
            <Box component="section" aria-labelledby="assessment-panel-title">
              {result ? <ResultCard result={result} architecture={result.architecture} comparisonLabel={comparisonLabel} onLabelChange={setComparisonLabel} onSave={saveComparison} /> : <CardPlaceholder title="Resilience Assessment" readiness={readiness} startupError={startupError} analysisError={analysisError} />}
              {analysisError && <Alert severity="error" sx={{ mt: 2 }} role="alert">{analysisError}</Alert>}
              <Button fullWidth variant="contained" size="large" startIcon={isAnalyzing ? <CircularProgress color="inherit" size={18} /> : <PlayArrowIcon />} disabled={!config || !isReady || isAnalyzing} onClick={handleAnalyze} sx={{ mt: 2, py: 1.5 }} data-testid="analyze-button">{isAnalyzing ? 'EVALUATING...' : 'ANALYZE ARCHITECTURE'}</Button>
            </Box>
          </Box>

          {result && <ComparisonPanel items={comparisons} onRemove={(id) => setComparisons((current) => current.filter((item) => item.id !== id))} onClear={() => setComparisons([])} />}
          {!result && <Paper component="section" aria-labelledby="system-context-placeholder" className="context-placeholder"><Typography id="system-context-placeholder" variant="h6">System Context</Typography><Typography variant="body2" color="text.secondary">Ground and integrated system service metrics will appear after a successful analysis.</Typography></Paper>}
        </Stack>
      </Container>
      <MethodologyDialog open={methodologyOpen} onClose={() => setMethodologyOpen(false)} />
    </ThemeProvider>
  )
}

const fallbackConfig: DSSConfig = {
  realization_count: 5,
  default_required_minimum_connectivity: 0.8,
  space_domain: { num_planes: { min: 4, max: 6 }, sats_per_plane: { min: 5, max: 8 }, altitude_km: { min: 300, max: 1200 }, inclination_deg: { min: 30, max: 98 }, satellite_node_failure_probability: { min: 0, max: 0.2 }, satellite_edge_failure_probability: { min: 0, max: 0.25 } },
  ground_domain: { ground_station_failure_probability: { min: 0, max: 0.4 } },
  model: { family: 'TGNN', target: 'space_gcc_fraction_original_min' },
}

function CardPlaceholder({ title, readiness, startupError, analysisError }: { title: string; readiness: Readiness | null; startupError: string | null; analysisError: string | null }) {
  return <Paper className="assessment-placeholder"><Typography variant="overline" className="eyebrow">Decision surface</Typography><Typography id="assessment-panel-title" variant="h5" component="h2">{title}</Typography><Typography color="text.secondary" sx={{ mt: 1.5 }}>{startupError ? 'Connect the UI to the SATNET API to load the authoritative contract.' : readiness?.status === 'NOT_READY' ? 'Analysis will be enabled when the service is ready.' : analysisError || 'Submit the configuration to evaluate expected minimum connectivity, margin, and system context.'}</Typography></Paper>
}
