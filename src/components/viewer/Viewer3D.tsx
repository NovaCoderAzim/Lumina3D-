import { Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, useGLTF, Grid } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import * as THREE from 'three';
import { DemoScene } from './DemoScene';
import { GLBModel, type ViewMode } from './GLBModel';
import { ModelErrorBoundary } from './ModelErrorBoundary';
import { GaussianSplatViewer } from './GaussianSplatViewer';
import type { AnalyticsResponse, SemanticCategory, SemanticObject } from '../../types';

interface Props {
  modelUrl: string | null;
  demoMode?: boolean;
  theme?: 'dark' | 'light';
  analytics?: AnalyticsResponse;
  objects: SemanticObject[];
  visibleCategories: Set<SemanticCategory>;
  selectedId: string | null;
  onSelect: (obj: SemanticObject) => void;
  measuring?: boolean;
  onMeasurePoint?: (p: THREE.Vector3) => void;
}

const DEFAULT_CAMERA: [number, number, number] = [11, 9, 12];

function Loader() {
  return (
    <mesh>
      <sphereGeometry args={[0.001]} />
      <meshBasicMaterial />
    </mesh>
  );
}

export function Viewer3D({
  modelUrl,
  demoMode = false,
  theme = 'dark',
  analytics,
  objects,
  visibleCategories,
  selectedId,
  onSelect,
  measuring,
  onMeasurePoint,
}: Props) {
  const controlsRef = useRef<OrbitControlsImpl>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const [pointSize, setPointSize] = useState<number>(0.022);
  const [wireframe, setWireframe] = useState<boolean>(false);
  const [showBase, setShowBase] = useState<boolean>(false);
  const [showInferred, setShowInferred] = useState<boolean>(false);
  const [pitch, setPitch] = useState<number>(-16.5);
  const [roll, setRoll] = useState<number>(0.0);
  const [groundY, setGroundY] = useState<number>(-2.2);
  const [gridOffset, setGridOffset] = useState<number>(0);
  const hasFramedUrlRef = useRef<string | null>(null);

  const surfaceAvailable = analytics?.surface_mesh_available === true;
  const denseAvailable = analytics?.dense_mvs_available === true || (analytics?.dense_points ?? 0) > 0;
  const textureAvailable = analytics?.texture_available === true;

  const currentProjectId = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('projectId') : null;
  const splatUrl = analytics?.splat_url || (currentProjectId ? `/api/projects/${currentProjectId}/splat/file` : null);
  const splatAvailable = analytics?.splat_available === true;

  // Automatically pick the highest quality representation on load (TEXTURED > HYBRID > DENSE > SPLATS > SPARSE)
  const [viewMode, setViewMode] = useState<ViewMode>(() => {
    if (analytics?.texture_available) return 'TEXTURED';
    if (surfaceAvailable) return 'HYBRID';
    if (denseAvailable) return 'DENSE';
    if (splatAvailable) return 'SPLATS';
    return 'SPARSE';
  });

  useEffect(() => {
    if (textureAvailable) {
      setViewMode('TEXTURED');
    } else if (surfaceAvailable) {
      setViewMode('HYBRID');
    } else if (denseAvailable) {
      setViewMode('DENSE');
    } else if (splatAvailable) {
      setViewMode('SPLATS');
    } else {
      setViewMode('SPARSE');
    }
  }, [surfaceAvailable, denseAvailable, textureAvailable, splatAvailable]);

  // Determine active model URL
  const activeUrl = modelUrl;

  const handleRetry = useCallback(() => {
    if (activeUrl) {
      try {
        useGLTF.clear(activeUrl);
      } catch {
        // ignore
      }
    }
    setLoadError(null);
    setRetryKey((k) => k + 1);
  }, [activeUrl]);

  const [resetKey, setResetKey] = useState<number>(0);

  const resetCamera = useCallback(() => {
    controlsRef.current?.reset();
    setResetKey((k) => k + 1);
  }, []);

  const toggleFullscreen = useCallback(() => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  }, []);

  useEffect(() => {
    hasFramedUrlRef.current = null;
  }, [activeUrl]);

  const handleFramed = useCallback((box: THREE.Box3) => {
    // Only frame camera once per unique modelUrl
    if (hasFramedUrlRef.current !== activeUrl && controlsRef.current) {
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z, 1);
      // Frame close-up to fill the screen with crisp, detailed geometry from the drone's perspective
      const dist = maxDim * 0.72;
      controlsRef.current.object.position.set(dist * 0.15, dist * 0.55, dist * 0.85);
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
      hasFramedUrlRef.current = activeUrl;
    }

    if (Number.isFinite(box.min.y)) {
      setGroundY(box.min.y - 0.01);
    }
  }, [activeUrl]);

  return (
    <div
      ref={containerRef}
      className="relative h-full w-full viewer-grid overflow-hidden"
      style={{ background: 'var(--viewer-bg-gradient)' }}
    >
      <ModelErrorBoundary
        fallback={(err, reset) => (
          <div className="absolute inset-0 flex items-center justify-center bg-[var(--void)] p-6 z-20">
            <div className="border border-[var(--err)]/40 bg-[var(--panel)] p-6 max-w-md text-center rounded shadow-2xl">
              <div className="mono text-xs text-[var(--err)] tracking-widest mb-2 font-bold">
                ⚠️ 3D GRAPHICS PIPELINE NOTICE
              </div>
              <p className="text-xs text-[var(--muted)] mb-4 leading-relaxed">
                {err.message || 'WebGL renderer encountered a temporary context or initialization issue.'}
              </p>
              <button
                onClick={() => {
                  handleRetry();
                  reset();
                }}
                className="px-5 py-2.5 bg-[var(--amber)] hover:bg-[var(--amber-bright)] text-[var(--void)] font-bold text-xs mono rounded transition cursor-pointer"
              >
                ↻ RELOAD 3D VIEWER
              </button>
            </div>
          </div>
        )}
      >
        <Canvas
          dpr={[1, 1.5]}
          gl={{
            powerPreference: 'high-performance',
            antialias: true,
            alpha: true,
            stencil: false,
            depth: true,
            toneMapping: THREE.ACESFilmicToneMapping,
            toneMappingExposure: theme === 'light' ? 1.05 : 1.25,
          }}
          onError={() => setLoadError('WebGL context failed to initialize')}
        >
          <PerspectiveCamera makeDefault position={DEFAULT_CAMERA} fov={45} />
          <OrbitControls
            ref={controlsRef}
            makeDefault
            enableDamping
            dampingFactor={0.05}
            target={[0, 0, 0]}
          />

          {/* Spatial 3D Ground Plane Grid */}
          <Grid
            position={[0, groundY + gridOffset, 0]}
            args={[120, 120]}
            cellSize={0.5}
            cellThickness={0.9}
            cellColor={theme === 'light' ? '#94a3b8' : '#283c50'}
            sectionSize={2.5}
            sectionThickness={1.4}
            sectionColor={theme === 'light' ? '#64748b' : '#3f5e7e'}
            fadeDistance={50}
            fadeStrength={1.2}
            infiniteGrid
          />

          {/* Studio Photogrammetry Lighting */}
          <ambientLight intensity={theme === 'light' ? 0.95 : 0.85} />
          <hemisphereLight
            args={theme === 'light' ? [0xffffff, 0xd0dbe5, 1.1] : [0xffffff, 0x334455, 1.35]}
            position={[0, 50, 0]}
          />
          <directionalLight
            position={[25, 45, 30]}
            intensity={theme === 'light' ? 1.2 : 1.15}
          />
          <directionalLight
            position={[-25, 30, -30]}
            intensity={0.85}
          />
          <directionalLight
            position={[0, -20, 0]}
            intensity={theme === 'light' ? 0.5 : 0.35}
          />

          <Suspense fallback={<Loader />}>
            {activeUrl && viewMode !== 'SPLATS' ? (
              <group
                onPointerDown={(e) => {
                  if (measuring && onMeasurePoint) {
                    e.stopPropagation();
                    onMeasurePoint(e.point.clone());
                  }
                }}
              >
                <ModelErrorBoundary
                  key={`${activeUrl}-${retryKey}`}
                  onError={(err) => {
                    console.error('Failed to load 3D model asset:', err);
                    setLoadError(err.message || 'Failed to fetch model GLB asset');
                  }}
                >
                  <GLBModel
                    url={activeUrl}
                    viewMode={viewMode}
                    pointSize={pointSize}
                    wireframe={wireframe}
                    showBase={showBase}
                    showInferred={showInferred}
                    pitch={pitch}
                    roll={roll}
                    onFramed={handleFramed}
                  />
                </ModelErrorBoundary>
              </group>
            ) : demoMode ? (
              <DemoScene
                objects={objects}
                visibleCategories={visibleCategories}
                selectedId={selectedId}
                onSelect={onSelect}
              />
            ) : null}
          </Suspense>
        </Canvas>
      </ModelErrorBoundary>

      {/* 3D Gaussian Splatting (3DGS) Radiance Field Renderer */}
      {viewMode === 'SPLATS' && splatUrl && (
        <GaussianSplatViewer
          splatUrl={splatUrl}
          pitch={pitch}
          roll={roll}
          theme={theme}
          groundY={groundY}
          gridOffset={gridOffset}
          resetKey={resetKey}
        />
      )}

      {/* Top Bar: View Mode Switcher & Provenance Controls */}
      <div className="absolute top-4 left-4 z-20 flex flex-wrap items-center gap-2 pointer-events-auto">
        <div className="flex rounded border border-[var(--line)] bg-[var(--panel)]/90 backdrop-blur text-xs mono overflow-hidden shadow-lg">
          <button
            onClick={() => splatAvailable && setViewMode('SPLATS')}
            disabled={!splatAvailable}
            title={!splatAvailable ? '3DGS unavailable' : 'Render continuous 3D Gaussian Splatting radiance field'}
            className={`px-3 py-1.5 transition-colors ${viewMode === 'SPLATS' && splatAvailable
                ? 'bg-amber-400/25 text-amber-300 font-bold border-r border-amber-400/30 shadow-inner'
                : splatAvailable
                  ? 'text-[var(--text)] hover:text-amber-300'
                  : 'text-[var(--muted-dim)] cursor-not-allowed opacity-40'
              }`}
          >
            ✨ 3DGS
          </button>
          <button
            onClick={() => setViewMode('OBSERVED')}
            title="Render authentic observed photogrammetric geometry only"
            className={`px-3 py-1.5 transition-colors border-l border-[var(--line)] ${viewMode === 'OBSERVED'
                ? 'bg-[var(--amber)]/20 text-[var(--amber)] font-bold'
                : 'text-[var(--muted)] hover:text-[var(--text)]'
              }`}
          >
            OBSERVED
          </button>
          <button
            onClick={() => denseAvailable && setViewMode('DENSE')}
            disabled={!denseAvailable}
            title={!denseAvailable ? 'Dense MVS unavailable' : 'Render dense point cloud'}
            className={`px-3 py-1.5 transition-colors border-l border-[var(--line)] ${viewMode === 'DENSE' && denseAvailable
                ? 'bg-[var(--amber)]/20 text-[var(--amber)] font-bold'
                : denseAvailable
                  ? 'text-[var(--muted)] hover:text-[var(--text)]'
                  : 'text-[var(--muted-dim)] cursor-not-allowed opacity-40'
              }`}
          >
            DENSE
          </button>
          <button
            onClick={() => surfaceAvailable && setViewMode('MESH')}
            disabled={!surfaceAvailable}
            title={!surfaceAvailable ? 'Mesh unavailable' : 'Render continuous surface mesh'}
            className={`px-3 py-1.5 transition-colors border-l border-[var(--line)] ${viewMode === 'MESH' && surfaceAvailable
                ? 'bg-[var(--amber)]/20 text-[var(--amber)] font-bold'
                : surfaceAvailable
                  ? 'text-[var(--muted)] hover:text-[var(--text)]'
                  : 'text-[var(--muted-dim)] cursor-not-allowed opacity-40'
              }`}
          >
            MESH
          </button>
          <button
            onClick={() => textureAvailable && setViewMode('TEXTURED')}
            disabled={!textureAvailable}
            title={!textureAvailable ? 'Texture map unavailable' : 'Render photogrammetric textured model'}
            className={`px-3 py-1.5 transition-colors border-l border-[var(--line)] ${viewMode === 'TEXTURED' && textureAvailable
                ? 'bg-[var(--amber)]/20 text-[var(--amber)] font-bold'
                : textureAvailable
                  ? 'text-[var(--muted)] hover:text-[var(--text)]'
                  : 'text-[var(--muted-dim)] cursor-not-allowed opacity-40'
              }`}
          >
            TEXTURED
          </button>
          <button
            onClick={() => setViewMode('INFERRED')}
            title="Inspect inferred structural completions"
            className={`px-3 py-1.5 transition-colors border-l border-[var(--line)] ${viewMode === 'INFERRED'
                ? 'bg-[var(--amber)]/20 text-[var(--amber)] font-bold'
                : 'text-[var(--muted)] hover:text-[var(--text)]'
              }`}
          >
            INFERRED
          </button>
          <button
            onClick={() => surfaceAvailable && setViewMode('HYBRID')}
            disabled={!surfaceAvailable}
            title={!surfaceAvailable ? 'Hybrid mode unavailable' : 'Composite observed surface + completions + micro-points'}
            className={`px-3 py-1.5 transition-colors border-l border-[var(--line)] ${viewMode === 'HYBRID' && surfaceAvailable
                ? 'bg-[var(--amber)]/20 text-[var(--amber)] font-bold'
                : surfaceAvailable
                  ? 'text-[var(--muted)] hover:text-[var(--text)]'
                  : 'text-[var(--muted-dim)] cursor-not-allowed opacity-40'
              }`}
          >
            HYBRID
          </button>
          <button
            onClick={() => setViewMode('PRESENTATION')}
            title="Complete scene with optional presentation base"
            className={`px-3 py-1.5 transition-colors border-l border-[var(--line)] ${viewMode === 'PRESENTATION'
                ? 'bg-[var(--amber)]/20 text-[var(--amber)] font-bold'
                : 'text-[var(--muted)] hover:text-[var(--text)]'
              }`}
          >
            PRESENTATION
          </button>
          <button
            onClick={() => setViewMode('SPARSE')}
            title="Sparse SfM camera tie-points"
            className={`px-3 py-1.5 transition-colors border-l border-[var(--line)] ${viewMode === 'SPARSE'
                ? 'bg-[var(--amber)]/20 text-[var(--amber)] font-bold'
                : 'text-[var(--muted)] hover:text-[var(--text)]'
              }`}
          >
            SPARSE
          </button>
        </div>

        {/* Inferred Completion Toggle */}
        <button
          onClick={() => setShowInferred((s) => !s)}
          title="Toggle inferred structural facade completions"
          className={`rounded border px-2.5 py-1 text-xs mono transition cursor-pointer shadow ${showInferred
              ? 'bg-[var(--amber)]/25 text-[var(--amber)] border-[var(--amber)] font-bold'
              : 'bg-[var(--panel)]/90 border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]'
            }`}
        >
          INFERRED: {showInferred ? 'ON' : 'OFF'}
        </button>

        {/* Optional Thin Presentation Base Toggle */}
        <button
          onClick={() => setShowBase((b) => !b)}
          title="Toggle optional thin presentation base plate (presentation-only, excluded from statistics)"
          className={`rounded border px-2.5 py-1 text-xs mono transition cursor-pointer shadow ${showBase || viewMode === 'PRESENTATION'
              ? 'bg-[var(--amber)]/25 text-[var(--amber)] border-[var(--amber)] font-bold'
              : 'bg-[var(--panel)]/90 border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]'
            }`}
        >
          BASE: {showBase || viewMode === 'PRESENTATION' ? 'ON' : 'OFF'}
        </button>

        {/* Wireframe Mode Toggle */}
        {(viewMode === 'MESH' || viewMode === 'HYBRID' || viewMode === 'OBSERVED' || viewMode === 'INFERRED' || viewMode === 'PRESENTATION') && (
          <button
            onClick={() => setWireframe((w) => !w)}
            title="Toggle topological wireframe overlay"
            className={`rounded border px-2.5 py-1 text-xs mono transition cursor-pointer shadow ${wireframe
                ? 'bg-[var(--amber)]/25 text-[var(--amber)] border-[var(--amber)] font-bold'
                : 'bg-[var(--panel)]/90 border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]'
              }`}
          >
            WIREFRAME: {wireframe ? 'ON' : 'OFF'}
          </button>
        )}

        {/* Point Size Controls */}
        {(viewMode === 'SPARSE' || viewMode === 'DENSE' || viewMode === 'HYBRID' || viewMode === 'PRESENTATION') && (
          <div className="flex items-center gap-1.5 rounded border border-[var(--line)] bg-[var(--panel)]/90 backdrop-blur px-2.5 py-1 text-xs mono shadow">
            <span className="text-[var(--muted)] text-[10px] tracking-wide">POINT SIZE</span>
            <button
              onClick={() => setPointSize((s) => Math.max(0.006, Number((s - 0.004).toFixed(3))))}
              title="Decrease point size"
              className="px-1.5 py-0.5 hover:bg-[var(--panel-raised)] text-[var(--text)] rounded cursor-pointer transition font-bold"
            >
              -
            </button>
            <span className="text-[var(--amber)] min-w-[34px] text-center font-semibold">{pointSize.toFixed(3)}</span>
            <button
              onClick={() => setPointSize((s) => Math.min(0.08, Number((s + 0.004).toFixed(3))))}
              title="Increase point size"
              className="px-1.5 py-0.5 hover:bg-[var(--panel-raised)] text-[var(--text)] rounded cursor-pointer transition font-bold"
            >
              +
            </button>
          </div>
        )}

        {/* Pitch Angle Alignment */}
        <div className="flex items-center gap-1.5 rounded border border-[var(--line)] bg-[var(--panel)]/90 backdrop-blur px-2.5 py-1 text-xs mono shadow">
          <span className="text-[var(--muted)] text-[10px] tracking-wide">PITCH</span>
          <button
            onClick={() => setPitch((p) => Number((p - 0.5).toFixed(1)))}
            title="Pitch front up / back down (-0.5°)"
            className="px-1.5 py-0.5 hover:bg-[var(--panel-raised)] text-[var(--text)] rounded cursor-pointer transition font-bold"
          >
            -
          </button>
          <span className="text-[var(--amber)] min-w-[42px] text-center font-semibold">
            {pitch > 0 ? `+${pitch.toFixed(1)}°` : `${pitch.toFixed(1)}°`}
          </span>
          <button
            onClick={() => setPitch((p) => Number((p + 0.5).toFixed(1)))}
            title="Pitch front down / back up (+0.5°)"
            className="px-1.5 py-0.5 hover:bg-[var(--panel-raised)] text-[var(--text)] rounded cursor-pointer transition font-bold"
          >
            +
          </button>
        </div>

        {/* Roll Angle Alignment */}
        <div className="flex items-center gap-1.5 rounded border border-[var(--line)] bg-[var(--panel)]/90 backdrop-blur px-2.5 py-1 text-xs mono shadow">
          <span className="text-[var(--muted)] text-[10px] tracking-wide">ROLL</span>
          <button
            onClick={() => setRoll((r) => Number((r - 0.5).toFixed(1)))}
            title="Roll tilt left (-0.5°)"
            className="px-1.5 py-0.5 hover:bg-[var(--panel-raised)] text-[var(--text)] rounded cursor-pointer transition font-bold"
          >
            -
          </button>
          <span className="text-[var(--amber)] min-w-[38px] text-center font-semibold">
            {roll > 0 ? `+${roll.toFixed(1)}°` : `${roll.toFixed(1)}°`}
          </span>
          <button
            onClick={() => setRoll((r) => Number((r + 0.5).toFixed(1)))}
            title="Roll tilt right (+0.5°)"
            className="px-1.5 py-0.5 hover:bg-[var(--panel-raised)] text-[var(--text)] rounded cursor-pointer transition font-bold"
          >
            +
          </button>
        </div>

        {/* Quick Level Preset */}
        <button
          onClick={() => {
            setPitch(-16.5);
            setRoll(0.0);
          }}
          title="Auto-level model flat to ground"
          className={`rounded border px-2.5 py-1 text-xs mono transition cursor-pointer shadow ${Math.abs(pitch - (-16.5)) < 0.1 && Math.abs(roll) < 0.1
              ? 'bg-[var(--amber)]/25 text-[var(--amber)] border-[var(--amber)] font-bold'
              : 'bg-[var(--panel)]/90 border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]'
            }`}
        >
          LEVEL
        </button>

        {/* Raw 0° Preset */}
        <button
          onClick={() => {
            setPitch(0.0);
            setRoll(0.0);
          }}
          title="Reset orientation to 0° raw"
          className={`rounded border px-2.5 py-1 text-xs mono transition cursor-pointer shadow ${pitch === 0.0 && roll === 0.0
              ? 'bg-[var(--amber)]/25 text-[var(--amber)] border-[var(--amber)] font-bold'
              : 'bg-[var(--panel)]/90 border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)]'
            }`}
        >
          RAW 0°
        </button>

        {/* Ground Grid Height Adjustment */}
        <div className="flex items-center gap-1.5 rounded border border-[var(--line)] bg-[var(--panel)]/90 backdrop-blur px-2.5 py-1 text-xs mono shadow">
          <span className="text-[var(--muted)] text-[10px] tracking-wide">GRID PLANE</span>
          <button
            onClick={() => setGridOffset((o) => Number((o - 0.08).toFixed(2)))}
            title="Lower grid plane"
            className="px-1.5 py-0.5 hover:bg-[var(--panel-raised)] text-[var(--text)] rounded cursor-pointer transition font-bold"
          >
            -
          </button>
          <span className="text-[var(--amber)] min-w-[36px] text-center font-semibold">
            {gridOffset > 0 ? `+${gridOffset.toFixed(2)}` : gridOffset.toFixed(2)}
          </span>
          <button
            onClick={() => setGridOffset((o) => Number((o + 0.08).toFixed(2)))}
            title="Raise grid plane"
            className="px-1.5 py-0.5 hover:bg-[var(--panel-raised)] text-[var(--text)] rounded cursor-pointer transition font-bold"
          >
            +
          </button>
        </div>

        {/* Provenance Badge */}
        <div className="rounded border border-[var(--line)] bg-[var(--panel)]/90 backdrop-blur px-2.5 py-1 text-[11px] mono text-[var(--muted)] shadow">
          <span>PROVENANCE: </span>
          <span className="text-[var(--amber)] font-semibold">
            {viewMode === 'SPLATS' ? '3D GAUSSIAN SPLATTING (3DGS) RADIANCE FIELD' :
              viewMode === 'OBSERVED' ? 'OBSERVED ONLY (100% PHOTOGRAMMETRY)' :
                viewMode === 'INFERRED' ? 'INFERRED COMPLETIONS ONLY' :
                  viewMode === 'SPARSE' ? 'OBSERVED SfM TIE-POINTS' :
                    viewMode === 'DENSE' ? 'OBSERVED MVS POINT CLOUD' :
                      viewMode === 'MESH' ? 'REFINED SURFACE MESH' :
                        viewMode === 'TEXTURED' ? 'PHOTOGRAMMETRIC TEXTURED' :
                          viewMode === 'PRESENTATION' ? 'PRESENTATION COMPOSITE' :
                            'HYBRID (OBSERVED + DENSE)'}
          </span>
          {analytics?.dense_points ? (
            <span className="text-[var(--muted)]"> · {(analytics.dense_points / 1e6).toFixed(1)}M pts</span>
          ) : null}
        </div>
      </div>

      {/* Insufficient Geometry / No Model Overlay for Real Projects */}
      {!modelUrl && !demoMode && (
        <div className="absolute inset-0 flex items-center justify-center bg-[var(--void)]/85 backdrop-blur-sm z-20">
          <div className="border border-[var(--amber)]/40 bg-[var(--panel)] p-6 max-w-md text-center shadow-2xl rounded">
            <div className="mono text-xs text-[var(--amber)] tracking-widest mb-2 font-bold flex items-center justify-center gap-2">
              <span>⚠️</span>
              <span>RECONSTRUCTION STATUS: INSUFFICIENT GEOMETRY</span>
            </div>
            <p className="text-xs text-[var(--text)] mt-2 leading-relaxed">
              COLMAP registered 0 camera poses from the uploaded video. Structure-from-Motion requires overlapping views from multiple angles to triangulate 3D coordinates.
            </p>
            <p className="text-[11px] text-[var(--muted)] mt-2 italic">
              Scientific Integrity Notice: No artificial or procedural placeholder geometry is displayed.
            </p>
          </div>
        </div>
      )}

      {/* Viewer Chrome Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2 z-10">
        <button
          onClick={resetCamera}
          title="Reset camera"
          className="w-9 h-9 flex items-center justify-center bg-[var(--panel)]/90 border border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)] hover:border-[var(--line-bright)] mono text-xs rounded"
        >
          ⟲
        </button>
        <button
          onClick={toggleFullscreen}
          title="Fullscreen"
          className="w-9 h-9 flex items-center justify-center bg-[var(--panel)]/90 border border-[var(--line)] text-[var(--muted)] hover:text-[var(--text)] hover:border-[var(--line-bright)] mono text-xs rounded"
        >
          {isFullscreen ? '⤡' : '⤢'}
        </button>
      </div>

      <div className="absolute bottom-4 left-4 mono text-[10px] tracking-widest text-[var(--muted-dim)] pointer-events-none">
        LEFT DRAG · ROTATE&nbsp;&nbsp;&nbsp;RIGHT DRAG · PAN&nbsp;&nbsp;&nbsp;SCROLL · ZOOM
      </div>

      {loadError && (
        <div className="absolute inset-0 flex items-center justify-center bg-[var(--void)]/90 backdrop-blur-sm z-30 pointer-events-auto">
          <div className="border border-[var(--err)]/40 bg-[var(--panel)] px-8 py-6 max-w-md text-center rounded-lg shadow-2xl">
            <div className="mono text-xs text-[var(--err)] tracking-widest mb-2 font-bold flex items-center justify-center gap-2">
              <span>⚠️</span>
              <span>MODEL LOAD NOTICE</span>
            </div>
            <p className="text-xs text-[var(--muted)] mb-4 leading-relaxed">
              {loadError.includes('502') || loadError.includes('Bad Gateway')
                ? 'Backend server temporarily responded with 502 Bad Gateway while connecting. Please click retry to establish connection.'
                : loadError}
            </p>
            <button
              onClick={handleRetry}
              className="px-5 py-2.5 bg-[var(--amber)] hover:bg-[var(--amber-bright)] text-[var(--void)] font-bold mono text-xs rounded transition cursor-pointer shadow-lg active:scale-95"
            >
              ↻ RETRY LOADING MODEL
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
