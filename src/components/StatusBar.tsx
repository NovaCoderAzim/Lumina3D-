import type { AnalyticsResponse } from '../types';

interface Props {
  analytics: AnalyticsResponse;
}

function qualityLabel(confScore: number | undefined, regRate: number): { label: string; color: string } {
  const score = confScore != null ? confScore * 100 : regRate;
  if (score >= 80) return { label: 'GOOD', color: 'var(--ok)' };
  if (score >= 50) return { label: 'FAIR', color: 'var(--warn)' };
  return { label: 'LOW CONFIDENCE', color: 'var(--err)' };
}

export function StatusBar({ analytics }: Props) {
  const quality = qualityLabel(analytics.confidence?.overall_confidence, analytics.registration_rate);
  const geomSource = analytics.geometry_source || (analytics.sparse_points > 0 ? 'SPARSE_SFM' : 'NONE');
  const scaleMode = analytics.metric_scale_available ? 'METRIC' : 'RELATIVE';

  return (
    <footer className="h-10 flex items-center gap-6 px-5 border-t border-[var(--line)] bg-[var(--panel)] mono text-xs overflow-x-auto">
      <span className="flex items-center gap-2">
        <span className="text-[var(--muted)]">Reconstruction:</span>
        <span style={{ color: quality.color }}>{quality.label}</span>
      </span>
      <span className="flex items-center gap-2">
        <span className="text-[var(--muted)]">Source:</span>
        <span className="text-[var(--text)]">{geomSource}</span>
      </span>
      <span className="flex items-center gap-2">
        <span className="text-[var(--muted)]">Registration:</span>
        <span className="text-[var(--text)]">{analytics.registration_rate.toFixed(1)}%</span>
      </span>
      <span className="flex items-center gap-2">
        <span className="text-[var(--muted)]">Scale:</span>
        <span className="text-[var(--amber)]">{scaleMode}</span>
      </span>
      <span className="flex items-center gap-2">
        <span className="text-[var(--muted)]">AI:</span>
        <span className="text-[var(--ok)]">READY</span>
      </span>
      <span className="ml-auto text-[var(--muted-dim)]">
        {analytics.objects} objects · {analytics.input_images} images
      </span>
    </footer>
  );
}
