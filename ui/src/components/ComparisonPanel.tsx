import { Alert, Button, Card, CardContent, IconButton, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Typography } from '@mui/material'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutline'
import type { SavedComparison } from '../types'
import { formatMargin, formatPercent } from '../format'

export default function ComparisonPanel({ items, onRemove, onClear }: { items: SavedComparison[]; onRemove: (id: string) => void; onClear: () => void }) {
  return (
    <Card component="section" aria-labelledby="comparison-title" className="panel-card comparison-card">
      <CardContent>
        <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ xs: 'flex-start', sm: 'center' }} gap={1}>
          <BoxTitle count={items.length} />
          {items.length > 0 && <Button size="small" color="inherit" onClick={onClear}>Clear comparisons</Button>}
        </Stack>
        {items.length === 0 ? <Alert severity="info" sx={{ mt: 2 }}>Save a successful analysis to compare architecture options here.</Alert> : <TableContainer component={Paper} variant="outlined" sx={{ mt: 2 }}><Table size="small" aria-label="Architecture comparison table"><TableHead><TableRow>{['Design', 'Satellites', 'Expected Min GCC', 'Requirement', 'Margin', 'Ground Service', 'Overall Service', 'Limiting Segment', 'Assessment', ''].map((heading) => <TableCell key={heading}>{heading}</TableCell>)}</TableRow></TableHead><TableBody>{items.map((item) => { const space = item.result.space_resilience; const context = item.result.system_context; const meets = space.expected_assessment === 'MEETS_EXPECTED_REQUIREMENT'; return <TableRow key={item.id}><TableCell><Typography fontWeight={700}>{item.label || 'Unnamed design'}</Typography></TableCell><TableCell>{item.architecture.num_planes * item.architecture.sats_per_plane}</TableCell><TableCell>{formatPercent(space.expected_minimum_gcc)}</TableCell><TableCell>{formatPercent(space.required_minimum_connectivity)}</TableCell><TableCell>{formatMargin(space.expected_margin)}</TableCell><TableCell>{formatPercent(context.mean_ground_service_fraction)}</TableCell><TableCell>{formatPercent(context.mean_overall_service_fraction)}</TableCell><TableCell>{context.limiting_segment || '—'}</TableCell><TableCell><Typography color={meets ? 'success.main' : 'error.main'} fontWeight={700}>{meets ? 'MEETS' : 'BELOW'}</Typography></TableCell><TableCell><IconButton aria-label={`Remove ${item.label}`} size="small" onClick={() => onRemove(item.id)}><DeleteOutlineIcon fontSize="small" /></IconButton></TableCell></TableRow>})}</TableBody></Table></TableContainer>}
      </CardContent>
    </Card>
  )
}

function BoxTitle({ count }: { count: number }) {
  return <div><Typography id="comparison-title" variant="h6" component="h2" className="section-title">Architecture Comparison</Typography><Typography variant="body2" color="text.secondary">{count}/4 saved designs · each design retains its own requirement threshold</Typography></div>
}
