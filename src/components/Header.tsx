import { useEffect, useState } from 'react';

interface Props {
  projectName: string;
  demoMode: boolean;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
  onNavigateHome?: () => void;
  onNewProject?: () => void;
}

export function Header({
  projectName,
  demoMode,
  theme = 'dark',
  onToggleTheme,
  onNavigateHome,
  onNewProject,
}: Props) {
  const [clock, setClock] = useState(() => new Date());

  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  return (
    <header className="h-14 flex items-center justify-between px-5 border-b border-[var(--line)] bg-[var(--panel)] transition-colors duration-200">
      <div className="flex items-center gap-3">
        {onNavigateHome ? (
          <button
            onClick={onNavigateHome}
            title="Return to Home & Projects"
            className="group flex items-center gap-2 mono font-bold text-sm tracking-wide hover:opacity-80 transition cursor-pointer"
          >
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-gradient-to-tr from-[var(--amber)] to-amber-300 shadow-sm" />
              Lumina<span className="text-[var(--amber)]">3D</span>
            </span>
            <span className="text-[11px] px-2 py-0.5 rounded bg-[var(--panel-raised)] text-[var(--muted)] border border-[var(--line)] group-hover:text-[var(--text)] group-hover:border-[var(--amber)]/50 transition">
              ← Projects
            </span>
          </button>
        ) : (
          <span className="mono font-bold text-sm tracking-wide flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-gradient-to-tr from-[var(--amber)] to-amber-300 shadow-sm" />
            Lumina<span className="text-[var(--amber)]">3D</span>
          </span>
        )}
        <span className="w-px h-4 bg-[var(--line)]" />
        <span className="mono text-xs text-[var(--muted)] truncate max-w-[180px] sm:max-w-[320px]">
          Project: <strong className="text-[var(--text)] font-medium">{projectName}</strong>
        </span>
      </div>

      <div className="flex items-center gap-2.5 sm:gap-3 mono text-[11px] text-[var(--muted)]">
        {/* Black & White / Theme Toggle Button */}
        {onToggleTheme && (
          <button
            onClick={onToggleTheme}
            title={`Switch to ${theme === 'dark' ? 'Light / White' : 'Dark / Black'} theme`}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded border border-[var(--line)] bg-[var(--panel-raised)] text-[var(--text)] hover:border-[var(--amber)] transition cursor-pointer text-xs mono shadow-sm"
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

        {onNewProject && (
          <button
            onClick={onNewProject}
            className="flex items-center gap-1.5 px-3 py-1 bg-[var(--amber)] text-black font-semibold text-xs tracking-wide rounded hover:brightness-110 active:scale-95 transition cursor-pointer shadow-sm"
          >
            <span>+</span>
            <span>New Project</span>
          </button>
        )}

        {demoMode && (
          <span className="hidden sm:flex items-center gap-1.5 text-[var(--amber)]">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--amber)] status-dot-live" />
            DEMO MODE
          </span>
        )}
        <span className="hidden md:inline">{clock.toLocaleTimeString([], { hour12: false })}</span>
      </div>
    </header>
  );
}
