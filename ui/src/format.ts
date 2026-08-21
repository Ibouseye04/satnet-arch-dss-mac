export const formatPercent = (fraction: number | null | undefined): string =>
  fraction === null || fraction === undefined || !Number.isFinite(fraction)
    ? '—'
    : `${Math.round(fraction * 100)}%`

export const formatMargin = (fraction: number): string => {
  const points = Math.round(fraction * 100)
  return `${points >= 0 ? '+' : ''}${points} percentage points`
}

export const abbreviatedHash = (value: string | undefined): string =>
  value ? `${value.slice(0, 12)}…` : '—'

export const safeBlockedReason = (reason: string | null | undefined): string | null => {
  if (!reason || /SATNET_DSS_|[A-Za-z]:\\|\/(?:Users|home|var)\//i.test(reason)) return null
  return reason
}
