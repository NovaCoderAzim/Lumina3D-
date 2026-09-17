import type { CaptureQuality } from '../../types';

interface Props {
  capture: CaptureQuality;
}

const VERDICT_COLOR: Record<string, string> = {
  GO: 'var(--ok)',
  MARGINAL: 'var(--warn)',
  NO_GO: 'var(--err)',
};

function Meter({ label, value }: { label: string; value: number }) {
  const color = value >= 65 ? 'var(--ok)' : value >= 45 ? 'var(--warn)' : 'var(--err)';
  return (
    <div className="py-1">
      <div className="flex justify-between mb-1">
        <span className="text-xs text-[var(--muted)]">{label}</span>
        <span className="mono text-xs" style={{ color }}>{value.toFixed(0)}</span>
      </div>
      <div className="h-1.5 bg-[var(--panel-raised)] rounded overflow-hidden">
        <div style={{ width: `${Math.min(100, value)}%`, background: color }} className="h-full" />
      </div>
    </div>
  );
}

// Shown while the reconstruction runs — gives the operator an immediate,
// instant assessment on whether the capture can reconstruct accurately.
export function CaptureQualityCard({ capture }: Props) {
  const color = VERDICT_COLOR[capture.verdict] ?? 'var(--muted)';
  return (
    <div className="mt-6 border border-[var(--line)] bg-[var(--panel)] rounded-sm p-5">
      <div className="flex items-center justify-between mb-3">
        <div className="mono text-xs tracking-[0.2em] text-[var(--muted)]">CAPTURE QUALITY</div>
        <div className="mono text-xs" style={{ color }}>
          {capture.verdict} · {capture.score.toFixed(0)}/100
        </div>
      </div>

      <Meter label="Sharpness (motion blur)" value={capture.sharpness_score} />
      <Meter label="Exposure consistency" value={capture.exposure_consistency} />
      <Meter label="Frame overlap" value={capture.overlap_score} />

      <div className="flex items-baseline justify-between py-1 mt-1">
        <span className="text-xs text-[var(--muted)]">Camera motion</span>
        <span className="mono text-xs text-[var(--text)]">
          {capture.motion_pattern} · ~{capture.coverage_estimate_deg.toFixed(0)}° coverage
        </span>
      </div>

      {capture.recommendations?.length > 0 && (
        <div className="mt-3 border-t border-[var(--line)] pt-2 space-y-1">
          {capture.recommendations.map((r, i) => (
            <p key={i} className="text-[11px] text-[var(--muted)] leading-snug">• {r}</p>
          ))}
        </div>
      )}
    </div>
  );
}
