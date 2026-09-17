import { useState, useRef, useEffect } from 'react';
import type { AnalyticsResponse, SemanticCategory } from '../types';
import { LayersPanel } from './layers/LayersPanel';
import { AnalyticsPanel } from './layers/AnalyticsPanel';

interface Props {
  analytics: AnalyticsResponse;
  visibleCategories: Set<SemanticCategory>;
  modelUrl: string | null;
  onToggleCategory: (cat: SemanticCategory) => void;
}

export function Sidebar({ analytics, visibleCategories, modelUrl, onToggleCategory }: Props) {
  const [showExportMenu, setShowExportMenu] = useState(false);
  const menuRef = useRef<HTMLDivElement>(null);

  const counts: Record<SemanticCategory, number> = {
    Vehicle: analytics.vehicles,
    Building: analytics.buildings,
    Person: analytics.people,
    Vegetation: analytics.vegetation,
  };

  // Close export menu when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (menuRef.current && !menuRef.current.contains(event.target as Node)) {
        setShowExportMenu(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const projectId = modelUrl?.match(/\/api\/projects\/([^/]+)/)?.[1];
  const apiBase = import.meta.env.VITE_API_BASE_URL || '';

  const downloadArtifact = (path: string, filename: string) => {
    const link = document.createElement('a');
    link.href = `${apiBase}${path}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setShowExportMenu(false);
  };

  return (
    <aside className="w-72 flex-shrink-0 border-r border-[var(--line)] bg-[var(--panel)] flex flex-col overflow-y-auto">
      <div className="p-4 space-y-6 flex-1">
        <LayersPanel visible={visibleCategories} counts={counts} onToggle={onToggleCategory} />
        <AnalyticsPanel analytics={analytics} />
      </div>

      <div className="p-4 border-t border-[var(--line)] relative" ref={menuRef}>
        <button
          onClick={() => setShowExportMenu(!showExportMenu)}
          className="w-full text-left px-3 py-2 text-xs font-semibold mono border border-[var(--line-bright)] bg-[var(--surface)] hover:border-[var(--amber)] hover:text-[var(--amber)] transition-all flex items-center justify-between shadow"
        >
          <span>EXPORT RECONSTRUCTION</span>
          <span className="text-[10px]">{showExportMenu ? '▲' : '▼'}</span>
        </button>

        {showExportMenu && (
          <div className="absolute bottom-16 left-4 right-4 bg-[var(--panel)] border border-[var(--line-bright)] rounded shadow-2xl z-50 p-2 space-y-1 backdrop-blur-md">
            <div className="text-[10px] mono text-[var(--muted)] px-2 py-1 tracking-wider uppercase border-b border-[var(--line)] mb-1">
              Download Artifacts
            </div>

            {projectId ? (
              <>
                <button
                  onClick={() => downloadArtifact(`/api/projects/${projectId}/model/file`, 'model.glb')}
                  className="w-full text-left px-2.5 py-1.5 text-xs text-[var(--text)] hover:bg-[var(--surface)] hover:text-[var(--amber)] rounded flex items-center justify-between transition-colors"
                >
                  <span className="mono">3D Model (.GLB)</span>
                  <span className="text-[10px] text-[var(--muted-dim)]">Mesh + Points</span>
                </button>

                <button
                  onClick={() => downloadArtifact(`/api/projects/${projectId}/mesh/file`, 'mesh.ply')}
                  className="w-full text-left px-2.5 py-1.5 text-xs text-[var(--text)] hover:bg-[var(--surface)] hover:text-[var(--amber)] rounded flex items-center justify-between transition-colors"
                >
                  <span className="mono">Surface Mesh (.PLY)</span>
                  <span className="text-[10px] text-[var(--muted-dim)]">Polygons</span>
                </button>

                <button
                  onClick={() => downloadArtifact(`/api/projects/${projectId}/dense/file`, 'cloud_dense.ply')}
                  className="w-full text-left px-2.5 py-1.5 text-xs text-[var(--text)] hover:bg-[var(--surface)] hover:text-[var(--amber)] rounded flex items-center justify-between transition-colors"
                >
                  <span className="mono">Dense Cloud (.PLY)</span>
                  <span className="text-[10px] text-[var(--muted-dim)]">3.4M Points</span>
                </button>

                <button
                  onClick={() => downloadArtifact(`/api/projects/${projectId}/report`, 'reconstruction_report.json')}
                  className="w-full text-left px-2.5 py-1.5 text-xs text-[var(--text)] hover:bg-[var(--surface)] hover:text-[var(--amber)] rounded flex items-center justify-between transition-colors"
                >
                  <span className="mono">Quality Report (.JSON)</span>
                  <span className="text-[10px] text-[var(--muted-dim)]">Metrics & Provenance</span>
                </button>
              </>
            ) : (
              <div className="p-2 text-xs text-[var(--muted)] italic">
                Export available for completed reconstruction projects.
              </div>
            )}
          </div>
        )}
      </div>
    </aside>
  );
}
