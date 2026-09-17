import { useState } from 'react';
import type { AnalyticsResponse } from '../../types';

interface Props {
  analytics: AnalyticsResponse;
  measuring: boolean;
  onToggleMeasure: () => void;
  lastDistanceUnits: number | null; // distance in model units, or null
}

// Measurement panel (SIH deliverable: "suitable for ... measurement").
// Converts model-unit distances to real metres using the GPS-recovered
// scale. If no metric scale is available, it says so honestly rather than
// showing a fake number.
export function MeasurePanel({ analytics, measuring, onToggleMeasure, lastDistanceUnits }: Props) {
  const scale = analytics.metric_scale?.metres_per_unit ?? null;
  const [note] = useState('Click two points on the model to measure.');

  const metres =
    lastDistanceUnits != null && scale != null ? lastDistanceUnits * scale : null;

  return (
    <div className="absolute bottom-16 left-4 w-64 border border-[var(--line)] bg-[var(--panel)]/95 rounded-sm p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="mono text-[11px] tracking-[0.2em] text-[var(--muted)]">MEASURE</span>
        <button
          onClick={onToggleMeasure}
          className="mono text-[10px] px-2 py-1 border transition-colors"
          style={{
            borderColor: measuring ? 'var(--amber)' : 'var(--line-bright)',
            color: measuring ? 'var(--amber)' : 'var(--muted)',
          }}
        >
          {measuring ? 'MEASURING · ON' : 'MEASURE · OFF'}
        </button>
      </div>

      {scale != null ? (
        <div className="text-[11px] text-[var(--muted)]">
          Scale: <span className="mono text-[var(--text)]">{scale.toFixed(3)} m/unit</span>
          {analytics.metric_scale?.is_synthetic_gps && (
            <span className="text-[var(--warn)]"> · synthetic GPS</span>
          )}
        </div>
      ) : (
        <div className="text-[11px] text-[var(--warn)]">
          No GPS scale — measurements unavailable (metric accuracy needs telemetry).
        </div>
      )}

      {measuring && (
        <p className="text-[10px] text-[var(--muted-dim)] mt-2 leading-snug">{note}</p>
      )}

      {metres != null && (
        <div className="mt-2 border-t border-[var(--line)] pt-2">
          <span className="text-[11px] text-[var(--muted)]">Distance: </span>
          <span className="mono text-sm text-[var(--amber)]">{metres.toFixed(2)} m</span>
        </div>
      )}
    </div>
  );
}
