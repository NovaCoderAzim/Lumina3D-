import { useEffect, useRef, useState } from 'react';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

interface Props {
  splatUrl: string;
  onClose?: () => void;
}

export function GaussianSplatViewer({ splatUrl }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!containerRef.current || !splatUrl) return;

    let isMounted = true;
    setLoading(true);
    setProgress(0);
    setError(null);

    // Patch document.body.removeChild to protect against @mkkellogg/gaussian-splats-3d bug
    // where it attempts document.body.removeChild(rootElement) when rootElement is a descendant.
    const originalBodyRemoveChild = document.body.removeChild.bind(document.body);
    document.body.removeChild = function <T extends Node>(child: T): T {
      if (child.parentNode === document.body) {
        return originalBodyRemoveChild(child);
      }
      if (child.parentNode) {
        child.parentNode.removeChild(child);
      }
      return child;
    };

    let viewerInstance: any = null;

    try {
      viewerInstance = new GaussianSplats3D.Viewer({
        rootElement: containerRef.current,
        cameraUp: [0, -1, 0], // In COLMAP photogrammetry world, -Y is UP towards sky
        initialCameraPosition: [0, -1.0, -14.0], // Positioned facing forward along drone flight axis
        initialCameraLookAt: [0, 0, 0], // Centered at origin
        sphericalHarmonicsDegree: 0,
        antialiased: true,
        gpuAcceleratedSort: false, // Standard WebGL sort prevents transform-feedback readback warnings
        sharedMemoryForWorkers: false,
        dynamicScene: false,
        showLoadingUI: false,
      });
      viewerRef.current = viewerInstance;

      viewerInstance
        .addSplatScene(splatUrl, {
          format: GaussianSplats3D.SceneFormat.Ply,
          progressiveLoad: false,
          showLoadingUI: false,
          splatAlphaRemovalThreshold: 1,
          position: [0, 0, -10.65], // Center the splats at (0, 0, 0)
          onProgress: (pct: number) => {
            if (isMounted) setProgress(Math.round(pct));
          },
        })
        .then(() => {
          if (isMounted) {
            setLoading(false);
            viewerInstance.start();
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
      // Restore original removeChild
      document.body.removeChild = originalBodyRemoveChild;
      viewerRef.current = null;
    };
  }, [splatUrl]);

  return (
    <div className="absolute inset-0 z-10 bg-black">
      <div ref={containerRef} className="w-full h-full" />

      {loading && (
        <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/80 z-20 pointer-events-none">
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

      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/90 z-20 p-6">
          <div className="border border-[var(--err)]/40 bg-[var(--panel)] p-6 max-w-md text-center rounded shadow-2xl">
            <div className="mono text-xs text-[var(--err)] tracking-widest mb-2 font-bold">
              ⚠️ 3DGS LOAD EXCEPTION
            </div>
            <p className="text-xs text-[var(--muted)] mb-4 leading-relaxed">{error}</p>
          </div>
        </div>
      )}

      {/* Floating HUD info */}
      <div className="absolute top-4 left-4 z-20 pointer-events-none">
        <div className="bg-black/60 backdrop-blur border border-[var(--line)] px-3 py-1.5 rounded text-[10px] mono text-[var(--amber)] flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
          <span>NEURAL RADIANCE FIELD / 3DGS ACTIVE • 60 FPS</span>
        </div>
      </div>
    </div>
  );
}
