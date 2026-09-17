import { useMemo } from 'react';
import type { SemanticCategory, SemanticObject } from '../../types';

interface Props {
  objects: SemanticObject[];
  visibleCategories: Set<SemanticCategory>;
  selectedId: string | null;
  onSelect: (obj: SemanticObject) => void;
}

// A restrained, low-poly "reconstructed site" built from primitives so the
// viewer has something meaningful to render without a real model.glb.
// Ground plane, a couple of buildings, roads, vehicles and tree clusters —
// positioned to line up with DEMO_OBJECTS in src/data/demoData.ts.
export function DemoScene({ objects, visibleCategories, selectedId, onSelect }: Props) {
  const byId = useMemo(() => new Map(objects.map((o) => [o.id, o])), [objects]);

  const clickable = (id: string) => byId.has(id);
  const handle = (id: string) => (e: any) => {
    e.stopPropagation();
    const obj = byId.get(id);
    if (obj) onSelect(obj);
  };

  const buildingVisible = visibleCategories.has('Building');
  const vehicleVisible = visibleCategories.has('Vehicle');
  const vegVisible = visibleCategories.has('Vegetation');

  return (
    <group>
      {/* ground */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[0, -0.01, 0]}>
        <planeGeometry args={[40, 40]} />
        <meshStandardMaterial color="#1a2129" roughness={1} />
      </mesh>

      {/* roads */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 4.1]}>
        <planeGeometry args={[12, 2.4]} />
        <meshStandardMaterial color="#0e1318" roughness={1} />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[1.5, 0.005, -1]}>
        <planeGeometry args={[2.4, 12]} />
        <meshStandardMaterial color="#0e1318" roughness={1} />
      </mesh>

      {/* buildings */}
      {buildingVisible && (
        <>
          <mesh
            position={[0, 1.1, -4]}
            castShadow
            onClick={clickable('bld-01') ? handle('bld-01') : undefined}
          >
            <boxGeometry args={[4.2, 2.2, 3]} />
            <meshStandardMaterial
              color={selectedId === 'bld-01' ? '#ffb020' : '#b0bec5'}
              emissive={selectedId === 'bld-01' ? '#3a2a05' : '#000000'}
            />
          </mesh>
          <mesh
            position={[6, 1.9, -1]}
            castShadow
            onClick={clickable('bld-02') ? handle('bld-02') : undefined}
          >
            <boxGeometry args={[2.6, 3.8, 2.6]} />
            <meshStandardMaterial
              color={selectedId === 'bld-02' ? '#ffb020' : '#8fa0aa'}
              emissive={selectedId === 'bld-02' ? '#3a2a05' : '#000000'}
            />
          </mesh>
        </>
      )}

      {/* vehicles */}
      {vehicleVisible && (
        <>
          <mesh position={[-3.2, 0.28, 4.1]} castShadow onClick={handle('veh-01')}>
            <boxGeometry args={[1.5, 0.5, 0.8]} />
            <meshStandardMaterial color={selectedId === 'veh-01' ? '#ffb020' : '#4fc3f7'} />
          </mesh>
          <mesh position={[-1.6, 0.28, 4.1]} castShadow onClick={handle('veh-02')}>
            <boxGeometry args={[1.5, 0.5, 0.8]} />
            <meshStandardMaterial color={selectedId === 'veh-02' ? '#ffb020' : '#4fc3f7'} />
          </mesh>
          <mesh position={[4.4, 0.4, -3.0]} castShadow onClick={handle('veh-03')}>
            <boxGeometry args={[2.1, 0.9, 1]} />
            <meshStandardMaterial color={selectedId === 'veh-03' ? '#ffb020' : '#3ea8d8'} />
          </mesh>
        </>
      )}

      {/* vegetation */}
      {vegVisible && (
        <>
          <group position={[5.5, 0, 4.5]} onClick={handle('veg-01')}>
            <mesh position={[0, 0.9, 0]} castShadow>
              <coneGeometry args={[0.9, 1.8, 8]} />
              <meshStandardMaterial color={selectedId === 'veg-01' ? '#ffb020' : '#7cb342'} />
            </mesh>
            <mesh position={[0.9, 0.6, 0.6]} castShadow>
              <coneGeometry args={[0.6, 1.2, 8]} />
              <meshStandardMaterial color={selectedId === 'veg-01' ? '#ffb020' : '#6fa438'} />
            </mesh>
          </group>
          <group position={[-5.5, 0, -1.5]} onClick={handle('veg-02')}>
            <mesh position={[0, 0.8, 0]} castShadow>
              <coneGeometry args={[0.8, 1.6, 8]} />
              <meshStandardMaterial color={selectedId === 'veg-02' ? '#ffb020' : '#7cb342'} />
            </mesh>
          </group>
        </>
      )}
    </group>
  );
}
