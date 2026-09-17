import { useCallback, useRef, useState } from 'react';

interface Props {
  onVideoReady: (file: File | null) => void;
  onBack?: () => void;
}

export function Upload({ onVideoReady, onBack }: Props) {
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = useCallback((files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    setFileName(f.name);
    setFile(f);
  }, []);

  return (
    <div className="h-full w-full flex flex-col items-center justify-center px-6 relative">
      {onBack && (
        <button
          onClick={onBack}
          className="absolute top-6 left-6 mono text-xs text-[var(--muted)] hover:text-[var(--text)] transition cursor-pointer flex items-center gap-1.5 px-3 py-1.5 rounded bg-[var(--panel)] border border-[var(--line)] hover:border-[var(--amber)]/50"
        >
          <span>← Back to All Projects</span>
        </button>
      )}

      <div className="w-full max-w-xl">
        <div className="mono text-xs tracking-[0.3em] text-[var(--muted)] mb-2">STEP 1 OF 2</div>
        <h2 className="text-2xl font-semibold mb-1">Upload Drone Video</h2>
        <p className="text-[var(--muted)] text-sm mb-8">
          A single continuous pass over the site. Slower movement and greater overlap between frames produce a more complete reconstruction.
        </p>

        <div
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); handleFiles(e.dataTransfer.files); }}
          onClick={() => inputRef.current?.click()}
          className={`cursor-pointer border-2 border-dashed rounded-sm p-12 text-center transition-colors ${
            dragging ? 'border-[var(--amber)] bg-[var(--amber-dim)]/10' : 'border-[var(--line-bright)] hover:border-[var(--muted)]'
          }`}
        >
          <input
            ref={inputRef}
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => handleFiles(e.target.files)}
          />
          <div className="mono text-4xl text-[var(--muted-dim)] mb-4">⤒</div>
          {fileName ? (
            <div>
              <div className="text-[var(--text)] font-medium text-lg">{fileName}</div>
              <div className="mono text-xs text-[var(--ok)] mt-2 font-semibold">
                ✓ Ready to upload — Click "Generate Digital Twin →" below
              </div>
              <div className="mono text-[11px] text-[var(--muted-dim)] mt-2">
                (Click inside this box if you wish to select a different video)
              </div>
            </div>
          ) : (
            <div>
              <div className="text-[var(--text)]">Drop a video file, or click to browse</div>
              <div className="mono text-xs text-[var(--muted-dim)] mt-2">MP4, MOV — single continuous pass</div>
            </div>
          )}
        </div>

        <div className="flex items-center justify-between mt-8">
          <button
            onClick={() => onVideoReady(null)}
            className="mono text-xs text-[var(--muted)] hover:text-[var(--text)] tracking-wide"
          >
            SKIP — USE DEMO PROJECT
          </button>
          <button
            disabled={!file}
            onClick={() => onVideoReady(file)}
            className={`px-6 py-2.5 bg-[var(--amber)] text-black font-semibold text-sm disabled:opacity-30 disabled:cursor-not-allowed hover:brightness-110 transition cursor-pointer ${
              file ? 'ring-2 ring-[var(--amber)] ring-offset-2 ring-offset-[var(--void)]' : ''
            }`}
          >
            Generate Digital Twin →
          </button>
        </div>
      </div>
    </div>
  );
}
