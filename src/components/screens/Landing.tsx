import { useEffect, useState, useMemo } from 'react';
import { api } from '../../lib/api';
import type { ProjectSummary } from '../../types';

interface Props {
  onCreateProject: () => void;
  onSelectProject: (projectId: string) => void;
  onResumeProject?: (projectId: string) => void;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
}

export function Landing({
  onCreateProject,
  onSelectProject,
  onResumeProject,
  theme = 'dark',
  onToggleTheme,
}: Props) {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterMode, setFilterMode] = useState<'all' | 'ready' | 'dense'>('all');

  const fetchProjects = async () => {
    try {
      setLoading(true);
      const res = await api.listProjects();
      if (res && Array.isArray(res.projects)) {
        setProjects(res.projects);
      }
    } catch (err) {
      console.warn('Could not fetch projects:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void fetchProjects();
  }, []);

  const filteredProjects = useMemo(() => {
    return projects.filter((p) => {
      const matchSearch =
        p.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        p.project_id.toLowerCase().includes(searchQuery.toLowerCase());
      if (!matchSearch) return false;

      if (filterMode === 'ready') return p.model_available || p.status === 'COMPLETED';
      if (filterMode === 'dense') return (p.dense_points ?? 0) > 0;
      return true;
    });
  }, [projects, searchQuery, filterMode]);

  const formatDate = (isoStr: string) => {
    try {
      const d = new Date(isoStr);
      return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        year: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return isoStr;
    }
  };

  return (
    <div className="h-full w-full flex flex-col bg-[var(--void)] text-[var(--text)] overflow-y-auto relative">
      {/* Background grid texture */}
      <div className="fixed inset-0 viewer-grid opacity-20 pointer-events-none" />

      {/* Top Header Bar */}
      <header className="sticky top-0 z-30 h-16 flex items-center justify-between px-6 md:px-10 border-b border-[var(--line)] bg-[var(--void)]/90 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <span className="mono font-bold text-base tracking-wider flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-sm bg-gradient-to-tr from-[var(--amber)] to-amber-300 shadow-sm" />
            Lumina<span className="text-[var(--amber)]">3D</span>
          </span>
          <span className="w-px h-4 bg-[var(--line)]" />
          <span className="mono text-xs text-[var(--muted)] hidden sm:inline">
            Photogrammetric & Neural Digital Twin Platform
          </span>
        </div>

        <div className="flex items-center gap-3">
          {onToggleTheme && (
            <button
              onClick={onToggleTheme}
              title={`Switch to ${theme === 'dark' ? 'Light / White' : 'Dark / Black'} theme`}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded border border-[var(--line)] bg-[var(--panel-raised)] text-[var(--text)] hover:border-[var(--amber)] transition cursor-pointer text-xs mono shadow-sm"
            >
              {theme === 'dark' ? (
                <>
                  <span className="text-amber-400 text-xs">☀️</span>
                  <span className="hidden sm:inline text-[10px] font-semibold text-[var(--muted)]">LIGHT</span>
                </>
              ) : (
                <>
                  <span className="text-indigo-600 text-xs">🌙</span>
                  <span className="hidden sm:inline text-[10px] font-semibold text-[var(--muted)]">DARK</span>
                </>
              )}
            </button>
          )}

          <button
            onClick={onCreateProject}
            className="group px-4 py-2 bg-[var(--amber)] text-black font-semibold text-xs mono tracking-wide rounded-sm hover:brightness-110 active:scale-95 transition cursor-pointer flex items-center gap-2 shadow-lg shadow-[var(--amber)]/10"
          >
            <span>+</span>
            <span>Start New Reconstruction</span>
            <span className="transition-transform group-hover:translate-x-0.5">→</span>
          </button>
        </div>
      </header>

      {/* Hero Section */}
      <section className="relative z-10 px-6 md:px-10 pt-10 pb-8 max-w-7xl mx-auto w-full">
        <div className="mono text-xs tracking-[0.3em] text-[var(--amber)] mb-3 flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-[var(--amber)] status-dot-live" />
          COMMAND CENTER · 3D RECONSTRUCTION ENGINE
        </div>

        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 pb-8 border-b border-[var(--line)]">
          <div className="max-w-2xl">
            <h1 className="text-3xl md:text-5xl font-bold tracking-tight text-[var(--text)]">
              Digital Twin <span className="text-[var(--amber)]">Mission Control</span>
            </h1>
            <p className="mt-3 text-[var(--muted)] text-sm md:text-base leading-relaxed">
              Select any previously generated 3D reconstruction below to immediately inspect the dense point cloud, continuous smooth surface mesh, and AI semantic layers — or launch a new drone video reconstruction.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={onCreateProject}
              className="px-6 py-3 bg-[var(--amber)] text-black font-semibold text-sm tracking-wide hover:brightness-110 active:scale-95 transition cursor-pointer flex items-center gap-2 shadow-lg"
            >
              <span>+ New Drone Project</span>
              <span>→</span>
            </button>
          </div>
        </div>

        {/* Feature Highlights Banner */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="bg-[var(--panel)] border border-[var(--line)] p-4 rounded-sm">
            <div className="mono text-[11px] text-[var(--muted)] uppercase tracking-wider">Dense MVS</div>
            <div className="mono text-xl font-bold text-[var(--amber)] mt-1">Multi-Million</div>
            <div className="text-xs text-[var(--muted-dim)] mt-0.5">Triangulated 3D Point Coordinates</div>
          </div>
          <div className="bg-[var(--panel)] border border-[var(--line)] p-4 rounded-sm">
            <div className="mono text-[11px] text-[var(--muted)] uppercase tracking-wider">Surface Mesh</div>
            <div className="mono text-xl font-bold text-[var(--ok)] mt-1">Continuous</div>
            <div className="text-xs text-[var(--muted-dim)] mt-0.5">Smooth Geometry & Real Texture</div>
          </div>
          <div className="bg-[var(--panel)] border border-[var(--line)] p-4 rounded-sm">
            <div className="mono text-[11px] text-[var(--muted)] uppercase tracking-wider">AI Semantics</div>
            <div className="mono text-xl font-bold text-[var(--text)] mt-1">Multi-Class</div>
            <div className="text-xs text-[var(--muted-dim)] mt-0.5">Vehicles, Buildings, Infrastructure</div>
          </div>
          <div className="bg-[var(--panel)] border border-[var(--line)] p-4 rounded-sm">
            <div className="mono text-[11px] text-[var(--muted)] uppercase tracking-wider">Metric Scale</div>
            <div className="mono text-xl font-bold text-[var(--text)] mt-1">Calibrated</div>
            <div className="text-xs text-[var(--muted-dim)] mt-0.5">Interactive 3D Distance Measurement</div>
          </div>
        </div>
      </section>

      {/* Projects List & Gallery Section */}
      <section className="relative z-10 px-6 md:px-10 pb-16 max-w-7xl mx-auto w-full flex-1">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="mono text-sm uppercase tracking-widest text-[var(--muted)] flex items-center gap-2">
              <span>Previous Reconstructions</span>
              <span className="px-2 py-0.5 rounded-full bg-[var(--panel-raised)] text-[var(--text)] text-xs border border-[var(--line)]">
                {projects.length}
              </span>
            </h2>
          </div>

          {/* Search and Filters */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="relative">
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search projects..."
                className="bg-[var(--panel)] border border-[var(--line)] text-xs px-3 py-1.5 rounded-sm focus:border-[var(--amber)] outline-none text-[var(--text)] w-44 sm:w-56"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-xs text-[var(--muted)] hover:text-[var(--text)]"
                >
                  ✕
                </button>
              )}
            </div>

            <div className="flex items-center border border-[var(--line)] rounded-sm bg-[var(--panel)] p-0.5 mono text-[11px]">
              <button
                onClick={() => setFilterMode('all')}
                className={`px-2.5 py-1 rounded-sm transition ${
                  filterMode === 'all'
                    ? 'bg-[var(--panel-raised)] text-[var(--amber)] font-medium'
                    : 'text-[var(--muted)] hover:text-[var(--text)]'
                }`}
              >
                All
              </button>
              <button
                onClick={() => setFilterMode('ready')}
                className={`px-2.5 py-1 rounded-sm transition ${
                  filterMode === 'ready'
                    ? 'bg-[var(--panel-raised)] text-[var(--amber)] font-medium'
                    : 'text-[var(--muted)] hover:text-[var(--text)]'
                }`}
              >
                3D Models Ready
              </button>
              <button
                onClick={() => setFilterMode('dense')}
                className={`px-2.5 py-1 rounded-sm transition ${
                  filterMode === 'dense'
                    ? 'bg-[var(--panel-raised)] text-[var(--amber)] font-medium'
                    : 'text-[var(--muted)] hover:text-[var(--text)]'
                }`}
              >
                Dense MVS
              </button>
            </div>

            <button
              onClick={() => void fetchProjects()}
              title="Refresh project list"
              className="p-1.5 rounded-sm bg-[var(--panel)] border border-[var(--line)] hover:border-[var(--amber)] text-[var(--muted)] hover:text-[var(--text)] transition cursor-pointer"
            >
              ↻
            </button>
          </div>
        </div>

        {/* Loading Skeleton */}
        {loading && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="bg-[var(--panel)] border border-[var(--line)] rounded-sm p-6 animate-pulse space-y-4"
              >
                <div className="h-4 bg-[var(--panel-raised)] w-1/3 rounded" />
                <div className="h-6 bg-[var(--panel-raised)] w-2/3 rounded" />
                <div className="h-16 bg-[var(--panel-raised)] rounded" />
                <div className="h-8 bg-[var(--panel-raised)] w-full rounded" />
              </div>
            ))}
          </div>
        )}

        {/* Empty State */}
        {!loading && filteredProjects.length === 0 && (
          <div className="border border-dashed border-[var(--line)] rounded-sm p-12 text-center bg-[var(--panel)]/40">
            <div className="mono text-3xl text-[var(--muted-dim)] mb-3">⬡</div>
            <h3 className="font-semibold text-lg text-[var(--text)]">No reconstructions found</h3>
            <p className="text-[var(--muted)] text-sm mt-1 max-w-md mx-auto">
              {searchQuery
                ? `No projects matching "${searchQuery}". Try clearing your search filter.`
                : 'Upload a drone flight video to produce your first high-precision 3D digital twin.'}
            </p>
            <button
              onClick={onCreateProject}
              className="mt-6 px-6 py-2.5 bg-[var(--amber)] text-black font-semibold text-xs mono tracking-wide rounded-sm hover:brightness-110 cursor-pointer"
            >
              + Upload Drone Video Now
            </button>
          </div>
        )}

        {/* Projects Cards Grid */}
        {!loading && filteredProjects.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {filteredProjects.map((proj) => {
              const hasDense = (proj.dense_points ?? 0) > 0;
              const hasSparse = (proj.sparse_points ?? 0) > 0;
              const isReady = proj.model_available || proj.status === 'COMPLETED';
              const isProcessing = proj.status === 'PROCESSING';
              const isFailed = proj.status === 'FAILED';

              return (
                <div
                  key={proj.project_id}
                  className="group relative bg-[var(--panel)] border border-[var(--line)] hover:border-[var(--amber)]/60 transition-all duration-200 rounded-sm p-5 flex flex-col justify-between shadow-lg hover:shadow-xl"
                >
                  <div>
                    {/* Top Status & Date */}
                    <div className="flex items-center justify-between gap-2 mb-3">
                      {isReady && (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] mono font-semibold bg-[var(--ok)]/10 text-[var(--ok)] border border-[var(--ok)]/30">
                          <span className="w-1.5 h-1.5 rounded-full bg-[var(--ok)]" />
                          3D MODEL READY
                        </span>
                      )}
                      {isProcessing && (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] mono font-semibold bg-[var(--amber)]/10 text-[var(--amber)] border border-[var(--amber)]/30">
                          <span className="w-1.5 h-1.5 rounded-full bg-[var(--amber)] animate-pulse" />
                          PROCESSING
                        </span>
                      )}
                      {isFailed && (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] mono font-semibold bg-[var(--err)]/10 text-[var(--err)] border border-[var(--err)]/30">
                          <span className="w-1.5 h-1.5 rounded-full bg-[var(--err)]" />
                          FAILED
                        </span>
                      )}
                      {!isReady && !isProcessing && !isFailed && (
                        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-[10px] mono text-[var(--muted)] bg-[var(--panel-raised)] border border-[var(--line)]">
                          {proj.status}
                        </span>
                      )}

                      <span className="mono text-[10px] text-[var(--muted-dim)]">
                        {formatDate(proj.created_at)}
                      </span>
                    </div>

                    {/* Project Title & ID */}
                    <h3 className="font-semibold text-lg text-[var(--text)] group-hover:text-[var(--amber)] transition-colors truncate">
                      {proj.name}
                    </h3>
                    <div className="mono text-xs text-[var(--muted)] mt-0.5 truncate">
                      {proj.project_id}
                    </div>

                    {/* Reconstruction Metrics Grid */}
                    <div className="grid grid-cols-2 gap-2 mt-4 pt-3 border-t border-[var(--line)] mono text-xs">
                      <div className="bg-[var(--panel-raised)]/70 p-2 rounded-sm">
                        <div className="text-[10px] text-[var(--muted)]">POINT CLOUD</div>
                        <div className="font-semibold text-[var(--text)] mt-0.5 truncate">
                          {hasDense ? (
                            <span className="text-[var(--amber)]">
                              {(proj.dense_points ?? 0).toLocaleString()} Dense
                            </span>
                          ) : hasSparse ? (
                            <span>{(proj.sparse_points ?? 0).toLocaleString()} SfM</span>
                          ) : (
                            <span className="text-[var(--muted-dim)]">—</span>
                          )}
                        </div>
                      </div>

                      <div className="bg-[var(--panel-raised)]/70 p-2 rounded-sm">
                        <div className="text-[10px] text-[var(--muted)]">CAMERAS</div>
                        <div className="font-semibold text-[var(--text)] mt-0.5 truncate">
                          {proj.registered_images > 0
                            ? `${proj.registered_images} Views`
                            : '—'}
                        </div>
                      </div>
                    </div>

                    {/* Feature Badges */}
                    <div className="flex flex-wrap gap-1.5 mt-3">
                      {proj.surface_mesh_available && (
                        <span className="mono text-[10px] px-1.5 py-0.5 rounded bg-[var(--ok)]/10 text-[var(--ok)] border border-[var(--ok)]/20">
                          ✓ Smooth Mesh
                        </span>
                      )}
                      {proj.dense_mvs_available && (
                        <span className="mono text-[10px] px-1.5 py-0.5 rounded bg-[var(--amber)]/10 text-[var(--amber)] border border-[var(--amber)]/20">
                          ✓ Dense MVS
                        </span>
                      )}
                      {proj.texture_available && (
                        <span className="mono text-[10px] px-1.5 py-0.5 rounded bg-[var(--panel-raised)] text-[var(--text)] border border-[var(--line)]">
                          ✓ Textured
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Action Button */}
                  <div className="mt-5 pt-3 border-t border-[var(--line)]">
                    {isReady ? (
                      <button
                        onClick={() => onSelectProject(proj.project_id)}
                        className="w-full py-2.5 px-4 bg-[var(--amber)] text-black font-semibold text-xs mono tracking-wide rounded-sm hover:brightness-110 active:scale-98 transition flex items-center justify-center gap-2 cursor-pointer shadow-md"
                      >
                        <span>Open 3D Digital Twin</span>
                        <span>→</span>
                      </button>
                    ) : isProcessing ? (
                      <button
                        onClick={() => onResumeProject ? onResumeProject(proj.project_id) : onSelectProject(proj.project_id)}
                        className="w-full py-2.5 px-4 bg-[var(--amber)]/20 text-[var(--amber)] border border-[var(--amber)]/40 font-semibold text-xs mono tracking-wide rounded-sm hover:bg-[var(--amber)]/30 transition flex items-center justify-center gap-2 cursor-pointer"
                      >
                        <span>View Progress</span>
                        <span>→</span>
                      </button>
                    ) : (
                      <button
                        onClick={() => onCreateProject()}
                        className="w-full py-2 px-4 bg-[var(--panel-raised)] text-[var(--muted)] hover:text-[var(--text)] border border-[var(--line)] hover:border-[var(--amber)] font-medium text-xs mono tracking-wide rounded-sm transition flex items-center justify-center gap-2 cursor-pointer"
                      >
                        <span>Start Over</span>
                        <span>↺</span>
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
