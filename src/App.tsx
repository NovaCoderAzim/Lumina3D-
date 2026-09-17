import { useCallback, useEffect, useRef, useState } from 'react';
import { Landing } from './components/screens/Landing';
import { Upload } from './components/screens/Upload';
import { Processing } from './components/screens/Processing';
import { DigitalTwin } from './components/screens/DigitalTwin';
import { DEMO_ANALYTICS, DEMO_OBJECTS, DEMO_PROJECT } from './data/demoData';
import { api } from './lib/api';
import {
  STAGE_ORDER,
  type AnalyticsResponse,
  type CaptureQuality,
  type ProcessingStage,
  type SemanticObject,
} from './types';

// Demo mode drives the full journey with precomputed data and no backend.
// Set VITE_DEMO_MODE=false (and VITE_API_BASE_URL) to use the real backend
// via src/lib/api.ts. Defaults to demo so the frontend runs standalone.
const DEMO_MODE = (import.meta.env.VITE_DEMO_MODE ?? 'true') !== 'false';

type Screen = 'landing' | 'upload' | 'processing' | 'twin';

const STAGE_DURATION_MS = 900;
const POLL_INTERVAL_MS = 1500;

interface FailureInfo {
  title: string;
  detail: string;
  cause?: string;
  suggestion?: string;
}

function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('lumina_theme');
      if (saved === 'light' || saved === 'dark') return saved;
    }
    return 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('lumina_theme', theme);
    } catch {
      // ignore
    }
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((t) => (t === 'dark' ? 'light' : 'dark'));
  }, []);

  const [screen, setScreen] = useState<Screen>('landing');
  const [stageIndex, setStageIndex] = useState(0);
  const [currentStage, setCurrentStage] = useState<ProcessingStage>(STAGE_ORDER[0]);
  const [failure, setFailure] = useState<FailureInfo | null>(null);
  const timer = useRef<number | null>(null);

  // Real-backend result state.
  const [projectName, setProjectName] = useState(DEMO_PROJECT.name);
  const [modelUrl, setModelUrl] = useState<string | null>(DEMO_PROJECT.model_url);
  const [objects, setObjects] = useState<SemanticObject[]>(DEMO_OBJECTS);
  const [analytics, setAnalytics] = useState<AnalyticsResponse>(DEMO_ANALYTICS);
  const [capture, setCapture] = useState<CaptureQuality | null>(null);
  const [progress, setProgress] = useState<number>(0);
  const [stageMessage, setStageMessage] = useState<string>('');

  // ---- Demo-mode synthetic stage timer (unchanged behaviour) ----
  useEffect(() => {
    if (DEMO_MODE === false) return;
    if (screen !== 'processing') return;
    if (stageIndex >= STAGE_ORDER.length - 1) {
      const t = window.setTimeout(() => setScreen('twin'), STAGE_DURATION_MS);
      return () => window.clearTimeout(t);
    }
    timer.current = window.setTimeout(() => setStageIndex((i) => i + 1), STAGE_DURATION_MS);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [screen, stageIndex]);

  useEffect(() => {
    setCurrentStage(STAGE_ORDER[Math.min(stageIndex, STAGE_ORDER.length - 1)]);
  }, [stageIndex]);

  // ---- Real backend flow: create -> upload -> poll -> results ----
  const runRealPipeline = useCallback(async (file: File | null) => {
    try {
      setFailure(null);
      setCapture(null);
      setProgress(5);
      setStageMessage('Uploading video & initializing...');
      setScreen('processing');
      setStageIndex(0);

      const name = `UAV-${new Date().toISOString().slice(0, 10)}`;
      setProjectName(name);
      const { project_id } = await api.createProject(name);

      // Upload the chosen video. A backend can also seed a demo video, so
      // if no file was chosen we still hit upload with an empty placeholder.
      if (file) {
        await api.uploadVideo(project_id, file);
      } else {
        // No file: create a tiny placeholder so the backend has something
        // to accept. The backend runs in mock mode regardless.
        const placeholder = new File([new Uint8Array([0])], 'demo.mp4', { type: 'video/mp4' });
        await api.uploadVideo(project_id, placeholder);
      }

      // Poll status until COMPLETED or FAILED.
      let capFetched = false;
      // eslint-disable-next-line no-constant-condition
      while (true) {
        const status = await api.getStatus(project_id);
        if (status.stage) setCurrentStage(status.stage);
        if (typeof status.progress === 'number') setProgress(status.progress);
        if (status.message) setStageMessage(status.message);

        // Capture quality is available very early (analyzed first). Fetch once.
        if (!capFetched) {
          try {
            const cq = await api.getCapture(project_id);
            if (cq && cq.verdict) {
              setCapture(cq);
              capFetched = true;
            }
          } catch {
            /* not ready yet */
          }
        }

        if (status.status === 'FAILED') {
          setFailure(
            status.error ?? {
              title: 'Processing failed',
              detail: 'The backend reported a failure without details.',
            },
          );
          return;
        }
        if (status.status === 'COMPLETED') break;
        await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
      }

      // Fetch results.
      const [info, sem, ana] = await Promise.all([
        api.getModelInfo(project_id),
        api.getSemantic(project_id),
        api.getAnalytics(project_id),
      ]);
      setModelUrl(info.available ? api.modelFileUrl(project_id) : null);
      setObjects(sem);
      setAnalytics(ana);
      setScreen('twin');
      // Persist projectId in browser URL so page reload never loses the generated model
      window.history.replaceState(null, '', `?projectId=${project_id}`);
    } catch (err) {
      setFailure({
        title: 'Cannot reach backend',
        detail: err instanceof Error ? err.message : String(err),
        suggestion:
          'Confirm the backend is running and VITE_API_BASE_URL points at it, ' +
          'or unset VITE_DEMO_MODE to use demo mode.',
      });
    }
  }, []);

  // Restore completed or in-progress project (e.g. from gallery click or ?projectId= URL)
  const loadExistingProject = useCallback(async (pid: string) => {
    try {
      setFailure(null);
      const status = await api.getStatus(pid).catch(() => ({
        project_id: pid,
        status: 'COMPLETED' as const,
        stage: null,
        progress: 100,
      }));

      setProjectName(`UAV-${pid}`);

      // If project is still processing, enter processing screen and poll
      if (status.status === 'PROCESSING') {
        if (status.stage) setCurrentStage(status.stage);
        if (typeof status.progress === 'number') setProgress(status.progress);
        if (status.message) setStageMessage(status.message);
        setScreen('processing');
        window.history.pushState(null, '', `?projectId=${pid}`);

        // Poll until done
        let capFetched = false;
        // eslint-disable-next-line no-constant-condition
        while (true) {
          const pollStatus = await api.getStatus(pid);
          if (pollStatus.stage) setCurrentStage(pollStatus.stage);
          if (typeof pollStatus.progress === 'number') setProgress(pollStatus.progress);
          if (pollStatus.message) setStageMessage(pollStatus.message);

          if (!capFetched) {
            try {
              const cq = await api.getCapture(pid);
              if (cq && cq.verdict) {
                setCapture(cq);
                capFetched = true;
              }
            } catch {
              /* not ready */
            }
          }

          if (pollStatus.status === 'FAILED') {
            setFailure(
              pollStatus.error ?? {
                title: 'Processing failed',
                detail: 'The reconstruction failed to complete.',
              }
            );
            return;
          }
          if (pollStatus.status === 'COMPLETED') break;
          await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        }
      }

      // Fetch completed results
      const [info, sem, ana] = await Promise.all([
        api.getModelInfo(pid).catch(() => ({ available: true, model_url: null, project_id: pid, format: 'glb' })),
        api.getSemantic(pid).catch(() => []),
        api.getAnalytics(pid).catch(() => DEMO_ANALYTICS),
      ]);
      setModelUrl(info.available ? api.modelFileUrl(pid) : null);
      setObjects(sem);
      setAnalytics(ana);
      setScreen('twin');
      window.history.pushState(null, '', `?projectId=${pid}`);
    } catch (err) {
      console.warn('Could not auto-restore project:', err);
    }
  }, []);

  const handleNavigateHome = useCallback(() => {
    setScreen('landing');
    window.history.pushState(null, '', window.location.pathname);
  }, []);

  const handleNewProject = useCallback(() => {
    setFailure(null);
    setCapture(null);
    setProgress(0);
    setStageMessage('');
    setScreen('upload');
    window.history.pushState(null, '', window.location.pathname);
  }, []);

  const handleSelectProject = useCallback(
    (pid: string) => {
      void loadExistingProject(pid);
    },
    [loadExistingProject]
  );

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const pid = params.get('projectId');
    if (pid) {
      void loadExistingProject(pid);
    }

    const handlePopState = () => {
      const p = new URLSearchParams(window.location.search);
      const curPid = p.get('projectId');
      if (curPid) {
        void loadExistingProject(curPid);
      } else {
        setScreen('landing');
      }
    };
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, [loadExistingProject]);

  const handleVideoReady = (file: File | null) => {
    if (file) {
      void runRealPipeline(file);
    } else if (DEMO_MODE) {
      setStageIndex(0);
      setScreen('processing');
    } else {
      void runRealPipeline(null);
    }
  };

  return (
    <div className="h-screen w-screen bg-[var(--void)] text-[var(--text)] overflow-hidden">
      {screen === 'landing' && (
        <Landing
          theme={theme}
          onToggleTheme={toggleTheme}
          onCreateProject={handleNewProject}
          onSelectProject={handleSelectProject}
          onResumeProject={handleSelectProject}
        />
      )}

      {screen === 'upload' && (
        <Upload onVideoReady={handleVideoReady} onBack={handleNavigateHome} />
      )}

      {screen === 'processing' && (
        <Processing
          currentStage={currentStage}
          progress={progress}
          message={stageMessage}
          failed={failure ?? undefined}
          capture={capture}
          onBackToHome={handleNavigateHome}
          onRetry={handleNewProject}
        />
      )}

      {screen === 'twin' && (
        <DigitalTwin
          projectName={projectName}
          demoMode={DEMO_MODE}
          theme={theme}
          onToggleTheme={toggleTheme}
          modelUrl={modelUrl}
          objects={objects}
          analytics={analytics}
          onNavigateHome={handleNavigateHome}
          onNewProject={handleNewProject}
        />
      )}
    </div>
  );
}

export default App;
