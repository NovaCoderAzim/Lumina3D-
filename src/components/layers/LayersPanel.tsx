import { CATEGORY_COLOR, CATEGORY_LAYER_LABEL, type SemanticCategory } from '../../types';

interface Props {
  visible: Set<SemanticCategory>;
  counts: Record<SemanticCategory, number>;
  onToggle: (cat: SemanticCategory) => void;
}

const CATEGORIES: SemanticCategory[] = ['Vehicle', 'Building', 'Person', 'Vegetation'];

export function LayersPanel({ visible, counts, onToggle }: Props) {
  return (
    <div>
      <div className="mono text-[11px] tracking-[0.2em] text-[var(--muted)] mb-3">SEMANTIC LAYERS</div>
      <div className="space-y-1">
        {CATEGORIES.map((cat) => {
          const active = visible.has(cat);
          return (
            <button
              key={cat}
              onClick={() => onToggle(cat)}
              className="w-full flex items-center gap-3 px-2 py-1.5 hover:bg-[var(--panel-raised)] transition-colors text-left"
            >
              <span
                className="w-3.5 h-3.5 border flex items-center justify-center flex-shrink-0"
                style={{
                  borderColor: active ? (CATEGORY_COLOR[cat] as string) : 'var(--line-bright)',
                  background: active ? (CATEGORY_COLOR[cat] as string) : 'transparent',
                }}
              >
                {active && <span className="w-1.5 h-1.5 bg-black" />}
              </span>
              <span className={`text-sm flex-1 ${active ? 'text-[var(--text)]' : 'text-[var(--muted)]'}`}>
                {CATEGORY_LAYER_LABEL[cat]}
              </span>
              <span className="mono text-xs text-[var(--muted-dim)]">{counts[cat] ?? 0}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
