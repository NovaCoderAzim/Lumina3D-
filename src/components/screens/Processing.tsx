import { useState, useEffect } from 'react';
import { STAGE_LABEL, STAGE_ORDER, type CaptureQuality, type ProcessingStage, type StageState } from '../../types';
import { CaptureQualityCard } from './CaptureQualityCard';

interface Props {
  currentStage: ProcessingStage;
  progress?: number;
  message?: string;
  failed?: { title: string; detail: string; cause?: string; suggestion?: string };
  capture?: CaptureQuality | null;
  onBackToHome?: () => void;
  onRetry?: () => void;
}

function stateFor(stage: ProcessingStage, current: ProcessingStage, failed: boolean): StageState {
  const currentIdx = STAGE_ORDER.indexOf(current);
  const stageIdx = STAGE_ORDER.indexOf(stage);
  if (failed && stageIdx === currentIdx) return 'failed';
  if (stageIdx < currentIdx) return 'completed';
  if (stageIdx === currentIdx) return 'processing';
  return 'pending';
}

const ICON: Record<StageState, string> = {
  completed: '✓',
  processing: '●',
  pending: '○',
  failed: '✕',
};

const COLOR: Record<StageState, string> = {
  completed: 'text-[var(--ok)]',
  processing: 'text-[var(--amber)]',
  pending: 'text-[var(--muted-dim)]',
  failed: 'text-[var(--err)]',
};

const TOTAL_EXPECTED_SECONDS = 102;

export function Processing({
  currentStage,
  progress,
  message,
  failed,
  capture,
  onBackToHome,
  onRetry,
}: Props) {
  const isFailed = !!failed;
  const [startTime] = useState(() => Date.now());
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Live timer tick tied to system wall clock - never freezes when switching tabs
  useEffect(() => {
    if (isFailed) return;
    const updateElapsed = () => {
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - startTime) / 1000)));
    };
    updateElapsed();
    const interval = window.setInterval(updateElapsed, 500);

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        updateElapsed();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      window.clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [startTime, isFailed]);

  // Derive dynamic progress:
  const stageIndex = Math.max(0, STAGE_ORDER.indexOf(currentStage));
  let basePct = 0;
  if (typeof progress === 'number' && progress > 0) {
    basePct = Math.min(99, Math.max(5, progress));
  } else {
    basePct = Math.round(((stageIndex + 1) / STAGE_ORDER.length) * 85);
  }

  // Smooth visual progress that crawls forward slightly during long stages
  const stageTimeElapsed = Math.max(0, elapsedSeconds - (stageIndex * 15));
  const timeBonus = Math.min(10, Math.floor(stageTimeElapsed / 4));
  const displayPct = isFailed ? 0 : Math.min(99, Math.max(basePct, basePct + timeBonus));

  // Dynamic remaining time calculation
  const formatTime = (secs: number) => {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const formatEta = () => {
    if (displayPct > 5 && elapsedSeconds > 10) {
      const estimatedTotal = Math.round(elapsedSeconds / (displayPct / 100));
      const remaining = Math.max(5, estimatedTotal - elapsedSeconds);
      const m = Math.floor(remaining / 60);
      const s = Math.floor(remaining % 60);
      if (m === 0) return `~${s}s remaining`;
      return `~${m}m ${s}s remaining`;
    }
    const typicalSeconds = 540;
    const remaining = Math.max(15, typicalSeconds - elapsedSeconds);
    const m = Math.floor(remaining / 60);
    const s = Math.floor(remaining % 60);
    if (m === 0) return `~${s}s remaining`;
    return `~${m}m ${s}s remaining`;
  };

  return (
    <div className="h-full w-full flex flex-col items-center justify-center px-6 overflow-y-auto py-8 relative">
      {onBackToHome && (
        <button
          onClick={onBackToHome}
          className="absolute top-6 left-6 mono text-xs text-[var(--muted)] hover:text-[var(--text)] transition cursor-pointer flex items-center gap-1.5 px-3 py-1.5 rounded bg-[var(--panel)] border border-[var(--line)] hover:border-[var(--amber)]/50"
        >
          <span>← Back to All Projects</span>
        </button>
      )}

      <div className="w-full max-w-xl">
        <div className="flex items-center justify-between mb-2">
          <div className="mono text-xs tracking-[0.3em] text-[var(--muted)]">STEP 2 OF 2 · RECONSTRUCTION ENGINE</div>
          {!isFailed && (
            <div className="flex items-center gap-3">
              <span className="mono text-xs text-[var(--muted)] flex items-center gap-1.5">
                <span className="inline-block w-2 h-2 rounded-full bg-[var(--ok)] animate-ping" />
                <span>Elapsed: {formatTime(elapsedSeconds)}</span>
              </span>
              <span className="mono text-xs text-[var(--amber)] bg-[var(--amber)]/10 px-2 py-0.5 rounded border border-[var(--amber)]/30 font-medium">
                ETA: {formatEta()}
              </span>
            </div>
          )}
        </div>

        <h2 className="text-2xl font-semibold mb-6 flex items-center justify-between">
          <span>{isFailed ? 'Reconstruction Failed' : 'Building Digital Twin'}</span>
          {!isFailed && (
            <span className="mono text-lg font-bold text-[var(--amber)]">
              {displayPct}%
            </span>
          )}
        </h2>

        {/* Live Animated Progress Bar */}
        {!isFailed && (
          <div className="mb-6 bg-[var(--panel)] border border-[var(--line)] rounded-sm p-4 shadow-lg">
            <div className="flex justify-between items-center mb-2.5">
              <span className="mono text-xs font-semibold text-[var(--text)] tracking-wide flex items-center gap-2">
                <span className="inline-block w-2 h-2 rounded-full bg-[var(--amber)] animate-pulse" />
                {message || (currentStage === 'RECONSTRUCTION' ? 'Structure-from-Motion triangulation & bundle adjustment...' : 'Processing stage...')}
              </span>
              <span className="mono text-xs text-[var(--muted)]">
                Stage {stageIndex + 1} of {STAGE_ORDER.length}
              </span>
            </div>

            {/* Glowing progress track */}
            <div className="relative h-3 bg-[var(--panel-raised)] overflow-hidden rounded-full border border-[var(--line)]">
              <div
                className="h-full bg-gradient-to-r from-[var(--amber-dim)] via-[var(--amber)] to-[var(--amber-bright)] transition-all duration-700 ease-out relative"
                style={{ width: `${displayPct}%` }}
              >
                {/* Subtle animated scan highlight */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/25 to-transparent animate-[shimmer_2s_infinite]" />
              </div>
            </div>

            {/* Hardware activity indicator */}
            <div className="mt-3 pt-2.5 border-t border-[var(--line)]/60 flex items-center justify-between text-[11px] mono text-[var(--muted)]">
              <div className="flex items-center gap-1.5">
                <span className="text-[var(--amber)]">⚡ GPU:</span>
                <span className="text-[var(--text)] font-medium">NVIDIA RTX 4060</span>
                <span className="text-[var(--muted-dim)]">· PyTorch CUDA Ready</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-[var(--ok)]">⚙️ CPU:</span>
                <span className="text-[var(--text)] font-medium">Intel i7 13th Gen</span>
                <span className="text-[var(--muted-dim)]">· Multi-core SfM</span>
              </div>
            </div>
          </div>
        )}

        {/* Stages Checklist */}
        <div className="bg-[var(--panel)] border border-[var(--line)] rounded-sm divide-y divide-[var(--line)] shadow-lg">
          {STAGE_ORDER.map((stage) => {
            const st = stateFor(stage, currentStage, isFailed);
            const isCurrent = stage === currentStage && !isFailed;
            return (
              <div
                key={stage}
                className={`flex items-start gap-4 p-4 transition-colors ${
                  isCurrent ? 'bg-[var(--panel-raised)]/60' : ''
                }`}
              >
                <span
                  className={`mono text-sm w-4 text-center mt-0.5 font-bold ${COLOR[st]} ${
                    st === 'processing' ? 'animate-pulse' : ''
                  }`}
                >
                  {ICON[st]}
                </span>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between">
                    <span
                      className={`text-sm font-medium ${
                        st === 'pending' ? 'text-[var(--muted-dim)]' : 'text-[var(--text)]'
                      }`}
                    >
                      {STAGE_LABEL[stage]}
                    </span>
                    {isCurrent && (
                      <span className="mono text-[10px] uppercase tracking-wider text-[var(--amber)] bg-[var(--amber)]/10 px-2 py-0.5 rounded border border-[var(--amber)]/30 font-semibold animate-pulse">
                        In Progress
                      </span>
                    )}
                  </div>
                  {isCurrent && (
                    <div className="mono text-xs text-[var(--amber)] mt-1 truncate">
                      {message || (stage === 'RECONSTRUCTION' ? 'Extracting SIFT descriptors and solving camera poses…' : 'Running module…')}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {capture && <div className="mt-6"><CaptureQualityCard capture={capture} /></div>}

        {isFailed && failed && (
          <div className="mt-6 border border-[var(--err)]/40 bg-[var(--err)]/5 rounded-sm p-5 shadow-lg">
            <div className="mono text-xs tracking-widest text-[var(--err)] mb-2 font-bold">{failed.title}</div>
            <p className="text-sm text-[var(--text)]">{failed.detail}</p>
            {failed.cause && (
              <p className="text-sm text-[var(--muted)] mt-2">
                <span className="text-[var(--muted)] font-semibold">Possible cause: </span>{failed.cause}
              </p>
            )}
            {failed.suggestion && (
              <p className="text-sm text-[var(--muted)] mt-1">
                <span className="text-[var(--muted)] font-semibold">Try: </span>{failed.suggestion}
              </p>
            )}

            <div className="flex items-center gap-3 mt-5 pt-4 border-t border-[var(--line)]">
              {onRetry && (
                <button
                  onClick={onRetry}
                  className="px-4 py-2 bg-[var(--amber)] text-black font-semibold text-xs rounded-sm hover:brightness-110 cursor-pointer"
                >
                  Upload New Video & Try Again
                </button>
              )}
              {onBackToHome && (
                <button
                  onClick={onBackToHome}
                  className="px-4 py-2 bg-[var(--panel-raised)] text-[var(--text)] border border-[var(--line)] font-medium text-xs rounded-sm hover:border-[var(--amber)]/50 cursor-pointer"
                >
                  Return to All Projects
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
