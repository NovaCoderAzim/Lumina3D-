import { useState } from 'react';
import * as THREE from 'three';
import { Header } from '../Header';
import { Sidebar } from '../Sidebar';
import { Viewer3D } from '../viewer/Viewer3D';
import { ObjectPanel } from '../ObjectPanel';
import { StatusBar } from '../StatusBar';
import { MeasurePanel } from '../layers/MeasurePanel';
import type { AnalyticsResponse, SemanticCategory, SemanticObject } from '../../types';

interface Props {
  projectName: string;
  demoMode: boolean;
  modelUrl: string | null;
  objects: SemanticObject[];
  analytics: AnalyticsResponse;
  theme?: 'dark' | 'light';
  onToggleTheme?: () => void;
  onNavigateHome?: () => void;
  onNewProject?: () => void;
}

const ALL_CATEGORIES: SemanticCategory[] = ['Vehicle', 'Building', 'Person', 'Vegetation'];

export function DigitalTwin({
  projectName,
  demoMode,
  modelUrl,
  objects,
  analytics,
  theme = 'dark',
  onToggleTheme,
  onNavigateHome,
  onNewProject,
}: Props) {
  const [visibleCategories, setVisibleCategories] = useState<Set<SemanticCategory>>(
    () => new Set(ALL_CATEGORIES.filter((c) => c !== 'Person')) // matches Section 5 mockup default (☐ People)
  );
  const [selected, setSelected] = useState<SemanticObject | null>(null);
  const [measuring, setMeasuring] = useState(false);
  const [measurePts, setMeasurePts] = useState<THREE.Vector3[]>([]);

  const lastDistanceUnits =
    measurePts.length === 2 ? measurePts[0].distanceTo(measurePts[1]) : null;

  const handleMeasurePoint = (p: THREE.Vector3) => {
    setMeasurePts((prev) => (prev.length >= 2 ? [p] : [...prev, p]));
  };

  const toggleCategory = (cat: SemanticCategory) => {
    setVisibleCategories((prev) => {
      const next = new Set(prev);
      next.has(cat) ? next.delete(cat) : next.add(cat);
      return next;
    });
  };

  return (
    <div className="h-full w-full flex flex-col">
      <Header
        projectName={projectName}
        demoMode={demoMode}
        theme={theme}
        onToggleTheme={onToggleTheme}
        onNavigateHome={onNavigateHome}
        onNewProject={onNewProject}
      />

      <div className="flex-1 flex min-h-0">
        <Sidebar
          analytics={analytics}
          visibleCategories={visibleCategories}
          modelUrl={modelUrl}
          onToggleCategory={toggleCategory}
        />

        <main className="flex-1 relative min-w-0">

          <Viewer3D
            modelUrl={modelUrl}
            demoMode={demoMode}
            theme={theme}
            analytics={analytics}
            objects={objects}
            visibleCategories={visibleCategories}
            selectedId={selected?.id ?? null}
            onSelect={setSelected}
            measuring={measuring}
            onMeasurePoint={handleMeasurePoint}
          />
          <ObjectPanel object={selected} onClose={() => setSelected(null)} />
          {modelUrl && (
            <MeasurePanel
              analytics={analytics}
              measuring={measuring}
              onToggleMeasure={() => { setMeasuring((m) => !m); setMeasurePts([]); }}
              lastDistanceUnits={lastDistanceUnits}
            />
          )}
        </main>
      </div>

      <StatusBar analytics={analytics} />
    </div>
  );
}
