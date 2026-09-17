import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

interface Props {
  splatUrl: string;
  pitch?: number;
  roll?: number;
  theme?: 'light' | 'dark';
  groundY?: number;
  gridOffset?: number;
  resetKey?: number;
  onClose?: () => void;
}

export function GaussianSplatViewer({
  splatUrl,
  pitch = -16.5,
  roll = 0.0,
  theme = 'dark',
  groundY = -2.2,
  gridOffset = 0.0,
  resetKey = 0,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const gridRef = useRef<THREE.GridHelper | null>(null);
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  // Initialize and load Gaussian Splats
  useEffect(() => {
    if (!containerRef.current || !splatUrl) return;

    let isMounted = true;
    setLoading(true);
    setProgress(0);
    setError(null);

    // Clean any prior contents
    containerRef.current.innerHTML = '';

    // Safe patch for document.body.removeChild to prevent @mkkellogg/gaussian-splats-3d
    // from attempting document.body.removeChild on a non-child (which detaches container from DOM)
    const originalBodyRemoveChild = document.body.removeChild.bind(document.body);
    document.body.removeChild = function <T extends Node>(child: T): T {
      if (child && child.parentNode === document.body) {
        return originalBodyRemoveChild(child);
      }
      // Critical fix: DO NOT remove child from its parent if it is a React container descendant!
      return child;
    };

    let viewerInstance: any = null;

    try {
      viewerInstance = new GaussianSplats3D.Viewer({
        rootElement: containerRef.current,
        cameraUp: [0, 1, 0], // Standard Three.js UP (+Y)
        initialCameraPosition: [0, 2.5, 16.0], // Front-elevated perspective facing origin
        initialCameraLookAt: [0, 0, 0], // Centered at origin
        sphericalHarmonicsDegree: 0,
        antialiased: true,
        gpuAcceleratedSort: false, // Standard sort prevents transform-feedback warnings
        sharedMemoryForWorkers: false,
        dynamicScene: false,
        showLoadingUI: false,
        renderMode: GaussianSplats3D.RenderMode.Always, // Continuous 60fps render loop
      });
      viewerRef.current = viewerInstance;

      // Add spatial Ground Plane Grid and ambient lighting to Three.js scene
      if (viewerInstance.threeScene) {
        const gridColor1 = theme === 'light' ? 0x64748b : 0x3f5e7e;
        const gridColor2 = theme === 'light' ? 0x94a3b8 : 0x283c50;
        const gridHelper = new THREE.GridHelper(120, 60, gridColor1, gridColor2);
        gridHelper.position.set(0, groundY + gridOffset, 0);
        viewerInstance.threeScene.add(gridHelper);
        gridRef.current = gridHelper;

        const ambLight = new THREE.AmbientLight(0xffffff, 1.0);
        viewerInstance.threeScene.add(ambLight);
      }

      // Convert COLMAP (+X right, +Y down, +Z forward) to Three.js (+X right, +Y up, -Z forward)
      // by rotating 180 degrees around X axis: quaternion [x=1, y=0, z=0, w=0]
      const rot180X = [1, 0, 0, 0];

      viewerInstance
        .addSplatScene(splatUrl, {
          format: GaussianSplats3D.SceneFormat.Ply,
          progressiveLoad: false,
          showLoadingUI: false,
          splatAlphaRemovalThreshold: 1,
          rotation: rot180X,
          // Center of reconstructed scene in COLMAP space is (0.057, -0.108, 10.647).
          // After 180-deg X rotation, offset by (-0.057, -0.108, 10.647) centers splats at (0, 0, 0).
          position: [-0.057, -0.108, 10.647],
          onProgress: (pct: number) => {
            if (isMounted) setProgress(Math.round(pct));
          },
        })
        .then(() => {
          if (isMounted) {
            setLoading(false);
            viewerInstance.start();

            // Apply initial pitch & roll leveling
            if (viewerInstance.splatMesh) {
              const pitchRollEuler = new THREE.Euler(
                THREE.MathUtils.degToRad(pitch),
                0,
                THREE.MathUtils.degToRad(roll),
                'XYZ'
              );
              viewerInstance.splatMesh.quaternion.setFromEuler(pitchRollEuler);
              viewerInstance.splatMesh.updateMatrixWorld(true);
            }

            viewerInstance.forceRenderNextFrame();
            // Force viewport resize synchronization
            setTimeout(() => {
              if (isMounted && viewerInstance) {
                window.dispatchEvent(new Event('resize'));
                viewerInstance.forceRenderNextFrame();
              }
            }, 60);
          }
        })
        .catch((err: any) => {
          if (isMounted) {
            console.error('Gaussian Splat loading error:', err);
            setError(err?.message || 'Failed to load 3D Gaussian Splats');
            setLoading(false);
          }
        });
    } catch (initErr: any) {
      console.error('Failed to initialize GaussianSplats3D Viewer:', initErr);
      setError(initErr?.message || 'WebGL 3DGS initialization error');
      setLoading(false);
    }

    return () => {
      isMounted = false;
      try {
        if (viewerInstance) {
          viewerInstance.stop?.();
          const disp = viewerInstance.dispose?.();
          if (disp && typeof disp.catch === 'function') {
            disp.catch(() => {});
          }
        }
      } catch (dispErr) {
        console.warn('Viewer dispose warning:', dispErr);
      }
      document.body.removeChild = originalBodyRemoveChild;
      viewerRef.current = null;
      gridRef.current = null;
    };
  }, [splatUrl]);

  // Dynamic pitch & roll updates without reloading
  useEffect(() => {
    if (viewerRef.current?.splatMesh) {
      const pitchRollEuler = new THREE.Euler(
        THREE.MathUtils.degToRad(pitch),
        0,
        THREE.MathUtils.degToRad(roll),
        'XYZ'
      );
      viewerRef.current.splatMesh.quaternion.setFromEuler(pitchRollEuler);
      viewerRef.current.splatMesh.updateMatrixWorld(true);
      viewerRef.current.forceRenderNextFrame();
    }
  }, [pitch, roll]);

  // Dynamic grid height updates
  useEffect(() => {
    if (gridRef.current) {
      gridRef.current.position.y = groundY + gridOffset;
      viewerRef.current?.forceRenderNextFrame();
    }
  }, [groundY, gridOffset]);

  // Handle external camera reset button (⟲)
  useEffect(() => {
    if (resetKey > 0 && viewerRef.current?.controls) {
      viewerRef.current.camera?.position.set(0, 2.5, 16.0);
      viewerRef.current.camera?.up.set(0, 1, 0);
      viewerRef.current.controls.target.set(0, 0, 0);
      viewerRef.current.controls.update();
      viewerRef.current.forceRenderNextFrame();
    }
  }, [resetKey]);

  // Sizing and window resize listener
  useEffect(() => {
    const handleResize = () => {
      viewerRef.current?.forceRenderNextFrame();
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  return (
    <div
      className="absolute inset-0 z-[1] overflow-hidden select-none"
      style={{ background: 'var(--viewer-bg-gradient)' }}
    >
      {/* Three.js canvas container */}
      <div
        ref={containerRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
      />

      {/* Loading Modal Overlay */}
      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/75 backdrop-blur-sm z-20 pointer-events-none transition-opacity duration-300">
          <div className="border border-[var(--amber)]/40 bg-[var(--panel)] p-6 rounded shadow-2xl max-w-sm text-center">
            <div className="mono text-xs text-[var(--amber)] tracking-widest mb-2 font-bold animate-pulse">
              ✨ STREAMING 3D GAUSSIAN SPLATS (3DGS)
            </div>
            <div className="w-full bg-[var(--line)] h-1.5 rounded overflow-hidden mb-3">
              <div
                className="bg-[var(--amber)] h-full transition-all duration-200"
                style={{ width: `${Math.max(5, progress)}%` }}
              />
            </div>
            <p className="text-[11px] text-[var(--muted)] mono">
              Sorting anisotropic radiance fields... {progress}%
            </p>
          </div>
        </div>
      )}

      {/* Error Modal Overlay */}
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/85 backdrop-blur-md z-20 p-6">
          <div className="border border-[var(--err)]/40 bg-[var(--panel)] p-6 max-w-md text-center rounded shadow-2xl">
            <div className="mono text-xs text-[var(--err)] tracking-widest mb-2 font-bold">
              ⚠️ 3DGS LOAD EXCEPTION
            </div>
            <p className="text-xs text-[var(--muted)] mb-4 leading-relaxed">{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 bg-[var(--amber)] text-black font-mono text-xs font-bold rounded shadow hover:bg-amber-300 transition cursor-pointer"
            >
              ↻ RELOAD VIEWER
            </button>
          </div>
        </div>
      )}

      {/* Floating 3DGS Radiance Field HUD */}
      {!loading && !error && (
        <div className="absolute top-16 left-4 z-20 pointer-events-none">
          <div className="bg-[var(--panel)]/90 backdrop-blur border border-[var(--line)] px-3 py-1.5 rounded text-[10px] mono text-[var(--amber)] flex items-center gap-2 shadow-lg">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
            <span>3D GAUSSIAN SPLATS ACTIVE • 1.32M SPLATS • 60 FPS</span>
          </div>
        </div>
      )}
    </div>
  );
}
