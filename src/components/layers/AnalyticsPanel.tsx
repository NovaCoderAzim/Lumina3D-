import type { AnalyticsResponse } from '../../types';

interface Props {
  analytics: AnalyticsResponse;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between py-1">
      <span className="text-xs text-[var(--muted)]">{label}</span>
      <span className="mono text-sm text-[var(--text)]">{value}</span>
    </div>
  );
}

function fmt(n: number) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return `${n}`;
}

export function AnalyticsPanel({ analytics }: Props) {
  const ms = analytics.metric_scale;
  const conf = analytics.confidence;
  const dm = analytics.dynamic_masking;
  const isRealGps = analytics.metric_scale_available && ms && !ms.is_synthetic_gps;

  return (
    <div className="space-y-5">
      <div>
        <div className="mono text-[11px] tracking-[0.2em] text-[var(--muted)] mb-2">RECONSTRUCTION</div>
        <div className="border-t border-[var(--line)] pt-1">
          <Row label="Geometry source" value={analytics.geometry_source || 'SPARSE_SFM'} />
          <Row label="Dense MVS engine" value={analytics.dense_engine || 'AUTO'} />
          <Row label="Surface mesh" value={analytics.surface_mesh_available ? 'AVAILABLE' : 'UNAVAILABLE'} />
          <Row label="Input images" value={`${analytics.input_images}`} />
          <Row label="Registered" value={`${analytics.registered_images}`} />
          <Row label="Registration rate" value={`${analytics.registration_rate.toFixed(1)}%`} />
          <Row label="Sparse points" value={fmt(analytics.sparse_points)} />
          <Row label="Dense points" value={fmt(analytics.dense_points)} />
          {analytics.reprojection_error != null && (
            <Row label="Reprojection error" value={`${analytics.reprojection_error.toFixed(2)} px`} />
          )}
        </div>
      </div>

      <div>
        <div className="mono text-[11px] tracking-[0.2em] text-[var(--muted)] mb-2">
          METRIC &amp; GEOREFERENCING
        </div>
        <div className="border-t border-[var(--line)] pt-1">
          {isRealGps ? (
            <>
              <Row label="Scale" value={`${ms.metres_per_unit.toFixed(3)} m/unit`} />
              <Row label="Alignment RMS" value={`${ms.rms_error_m.toFixed(2)} m`} />
              {ms.anchor_lat != null && ms.anchor_lon != null && (
                <Row label="Geo-anchor" value={`${ms.anchor_lat.toFixed(4)}, ${ms.anchor_lon.toFixed(4)}`} />
              )}
              <Row label="Telemetry" value={analytics.telemetry_source || 'REAL_GPS'} />
            </>
          ) : (
            <>
              <Row label="Metric scale" value="UNSCALED / RELATIVE" />
              <Row label="GPS telemetry" value={analytics.gps_available ? 'AVAILABLE' : 'NOT AVAILABLE'} />
              <Row label="Georeferenced" value={analytics.georeferenced ? 'YES' : 'NO'} />
              <p className="text-[10px] text-[var(--muted-dim)] mt-1.5 italic leading-relaxed">
                Video-only input. Metric scale requires synchronized GPS, RTK, or surveyed control points.
              </p>
            </>
          )}
        </div>
      </div>

      {conf && (
        <div>
          <div className="mono text-[11px] tracking-[0.2em] text-[var(--muted)] mb-2">CONFIDENCE &amp; COVERAGE</div>
          <div className="border-t border-[var(--line)] pt-1">
            <Row label="Overall confidence" value={`${Math.round(conf.overall_confidence * 100)}%`} />
            <Row label="Coverage" value={`${Math.round(conf.coverage_score * 100)}%`} />
            <div className="flex h-2 rounded overflow-hidden my-2">
              <div style={{ width: `${conf.high_conf_fraction * 100}%`, background: 'var(--ok)' }} title="high" />
              <div style={{ width: `${conf.medium_conf_fraction * 100}%`, background: 'var(--warn)' }} title="medium" />
              <div style={{ width: `${conf.low_conf_fraction * 100}%`, background: 'var(--err)' }} title="low" />
            </div>
            <Row label="Weak regions flagged" value={`${conf.weak_region_count}`} />
          </div>
          {conf.notes && <p className="text-[10px] text-[var(--muted-dim)] mt-1 leading-snug">{conf.notes}</p>}
        </div>
      )}

      {dm && (
        <div>
          <div className="mono text-[11px] tracking-[0.2em] text-[var(--muted)] mb-2">DYNAMIC OBJECT MASKING</div>
          <div className="border-t border-[var(--line)] pt-1">
            <Row label="Dynamic detections" value={`${dm.dynamic_detections}`} />
            <Row label="Frames masked" value={`${dm.images_masked}/${dm.total_images}`} />
            <Row label="Pixels excluded" value={`${(dm.mean_masked_fraction * 100).toFixed(1)}%`} />
          </div>
        </div>
      )}

      <div>
        <div className="mono text-[11px] tracking-[0.2em] text-[var(--muted)] mb-2">AI PERCEPTION (2D → 3D)</div>
        <div className="border-t border-[var(--line)] pt-1">
          <Row label="Detector" value="YOLOv8 GPU" />
          <Row label="Detected objects" value={`${analytics.objects}`} />
          <Row label="Vehicles" value={`${analytics.vehicles}`} />
          <Row label="People" value={`${analytics.people}`} />
          {analytics.buildings > 0 && <Row label="Buildings" value={`${analytics.buildings}`} />}
          {analytics.vegetation > 0 && <Row label="Vegetation" value={`${analytics.vegetation}`} />}
          <Row label="Avg confidence" value={`${Math.round(analytics.average_detection_confidence * 100)}%`} />
        </div>
      </div>

      {analytics.warnings && analytics.warnings.length > 0 && (
        <div className="p-2.5 rounded bg-[var(--surface)] border border-[var(--line)]">
          <div className="mono text-[10px] text-[var(--amber)] font-bold mb-1">PROVENANCE NOTICES</div>
          <ul className="text-[10px] text-[var(--muted)] space-y-1 list-disc pl-3">
            {analytics.warnings.map((w, idx) => (
              <li key={idx}>{w}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
