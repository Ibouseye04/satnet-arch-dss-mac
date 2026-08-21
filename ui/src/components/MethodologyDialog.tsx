import { Dialog, DialogContent, DialogTitle, Divider, IconButton, List, ListItem, ListItemText, Typography } from '@mui/material'
import CloseIcon from '@mui/icons-material/Close'

export default function MethodologyDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  return <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth aria-labelledby="methodology-title"><DialogTitle id="methodology-title">Methodology & About<IconButton aria-label="Close methodology" onClick={onClose} sx={{ position: 'absolute', right: 8, top: 8 }}><CloseIcon /></IconButton></DialogTitle><DialogContent dividers>
    <Typography variant="h6" gutterBottom>How SATNET evaluates an architecture</Typography>
    <Typography variant="body2" color="text.secondary" paragraph>SATNET evaluates proposed satellite architectures using orbital/network simulation, five deterministic failure realizations, frozen temporal graph neural network regression, deterministic ground/system service calculations, and comparison against a user-defined engineering requirement.</Typography>
    <List dense>
      <ListItem><ListItemText primary="Expected Minimum Connectivity" secondary="The arithmetic mean of the five TGNN-predicted minimum GCC values." /></ListItem>
      <ListItem><ListItemText primary="Meeting count" secondary="The 5-realization meeting count is descriptive and is not a calibrated probability." /></ListItem>
      <ListItem><ListItemText primary="Metric provenance" secondary="TGNN predicts space-segment resilience. Ground and integrated system metrics are SATNET-calculated values." /></ListItem>
    </List>
    <Divider sx={{ my: 2 }} />
    <Typography variant="h6" gutterBottom>Model limitations</Typography>
    <List dense>
      <ListItem><ListItemText primary="Validated domain" secondary="The model is restricted to validated architecture ranges supplied by the DSS API." /></ListItem>
      <ListItem><ListItemText primary="Prediction target" secondary="Minimum original-denominator GCC." /></ListItem>
      <ListItem><ListItemText primary="Temporal sequence" secondary="The model operates on an 11-timestep temporal sequence." /></ListItem>
      <ListItem><ListItemText primary="Engineering scenarios" secondary="Five realizations are deterministic engineering scenarios, not a Monte Carlo probability estimate." /></ListItem>
      <ListItem><ListItemText primary="External evaluation" secondary="External evaluation used real Starlink orbital observations transformed through SATNET methodology, not proprietary Starlink routing or service telemetry." /></ListItem>
      <ListItem><ListItemText primary="Ground/system values" secondary="Ground and integrated system values are calculated rather than TGNN-predicted." /></ListItem>
    </List>
    <Divider sx={{ my: 2 }} />
    <Typography variant="body2" color="text.secondary">The operational TGNN was selected because continuous resilience-regression performance materially exceeded the simpler Random Forest benchmark, while classification performance was comparable. Random Forest remains part of the dissertation narrative.</Typography>
  </DialogContent></Dialog>
}
