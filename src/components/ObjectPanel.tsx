import type { ReactNode } from 'react';
import { CATEGORY_COLOR, type SemanticObject } from '../types';

interface Props {
  object: SemanticObject | null;
  onClose: () => void;
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="py-2 border-b border-[var(--line)]">
      <div className="mono text-[10px] tracking-[0.2em] text-[var(--muted-dim)]">{label}</div>
      <div className="text-sm text-[var(--text)] mt-0.5">{children}</div>
    </div>
  );
}

export function ObjectPanel({ object, onClose }: Props) {
  if (!object) return null;

  const hasPosition = object.estimated_3d_position !== null;

  return (
    <div className="absolute top-4 left-4 w-72 bg-[var(--panel)]/95 border border-[var(--line)] backdrop-blur-sm">
      <div className="flex items-center justify-between px-4 py-3 border-b border-[var(--line)]">
        <div className="flex items-center gap-2">
          <span
            className="w-2.5 h-2.5 rounded-full"
            style={{ background: CATEGORY_COLOR[object.category] as string }}
          />
          <span className="mono text-xs tracking-[0.2em] text-[var(--muted)]">OBJECT</span>
        </div>
        <button onClick={onClose} className="text-[var(--muted)] hover:text-[var(--text)] text-sm leading-none">
          ✕
        </button>
      </div>

      <div className="px-4 pb-3">
        <Field label="ID">{object.id}</Field>
        <Field label="Class">{object.class}</Field>
        <Field label="Semantic category">{object.category}</Field>
        <Field label="Detection confidence">{Math.round(object.confidence * 100)}%</Field>
        <Field label="Observations">{object.observations} frames</Field>
        <div className="py-2">
          <div className="mono text-[10px] tracking-[0.2em] text-[var(--muted-dim)]">3D ASSOCIATION</div>
          {hasPosition ? (
            <div className="mono text-sm text-[var(--ok)] mt-0.5">Available</div>
          ) : (
            <div className="mono text-sm text-[var(--muted)] mt-0.5">Unavailable</div>
          )}
        </div>
      </div>
    </div>
  );
}
