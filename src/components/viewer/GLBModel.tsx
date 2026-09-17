import { useEffect, useMemo, useRef } from 'react';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

export type ViewMode = 'SPLATS' | 'OBSERVED' | 'DENSE' | 'MESH' | 'TEXTURED' | 'INFERRED' | 'HYBRID' | 'PRESENTATION' | 'SPARSE';

interface Props {
  url: string;
  viewMode?: ViewMode;
  pointSize?: number;
  wireframe?: boolean;
  showBase?: boolean;
  showInferred?: boolean;
  pitch?: number;
  roll?: number;
  onFramed?: (box: THREE.Box3) => void;
}

// Module-level cached circular radial alpha texture for soft photogrammetric splats.
let cachedPointTexture: THREE.Texture | null = null;
function getPointAlphaTexture(): THREE.Texture {
  if (!cachedPointTexture) {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const grad = ctx.createRadialGradient(32, 32, 0, 32, 32, 32);
      grad.addColorStop(0, 'rgba(255,255,255,1)');
      grad.addColorStop(0.7, 'rgba(255,255,255,0.95)');
      grad.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, 64, 64);
    }
    cachedPointTexture = new THREE.CanvasTexture(canvas);
    cachedPointTexture.needsUpdate = true;
  }
  return cachedPointTexture;
}

export function GLBModel({
  url,
  viewMode = 'HYBRID',
  pointSize = 0.022,
  wireframe = false,
  showBase = false,
  showInferred = true,
  pitch = -16.5,
  roll = 0.0,
  onFramed,
}: Props) {
  const { scene } = useGLTF(url);
  const pivotRef = useRef<THREE.Group>(null);

  // Initialize scene and configure photogrammetric materials once per model
  const { modelGroup } = useMemo(() => {
    const group = scene.clone(true);
    const pointTexture = getPointAlphaTexture();

    group.traverse((child) => {
      if (child instanceof THREE.Points) {
        child.frustumCulled = true;
        child.material = new THREE.PointsMaterial({
          size: pointSize,
          sizeAttenuation: true,
          vertexColors: true,
          map: pointTexture,
          transparent: true,
          alphaTest: 0.02,
          depthWrite: true,
          depthTest: true,
        });
      } else if (child instanceof THREE.Mesh) {
        child.frustumCulled = true;
        const name = (child.name || '').toLowerCase();
        const isBase = name.includes('presentation_base');
        const isInferred = name.includes('inferred') || name.includes('completion');

        try {
          child.geometry.computeVertexNormals();
        } catch {
          /* ignore */
        }

        if (isBase) {
          // Matte charcoal presentation base
          child.material = new THREE.MeshStandardMaterial({
            color: 0x24282e,
            roughness: 0.95,
            metalness: 0.02,
            side: THREE.DoubleSide,
          });
        } else if (isInferred) {
          // Inferred completion material: subtle semi-transparent warm tint
          child.material = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.85,
            metalness: 0.02,
            side: THREE.DoubleSide,
            transparent: true,
            opacity: 0.92,
            polygonOffset: true,
            polygonOffsetFactor: 1.0,
            polygonOffsetUnits: 1.0,
          });
        } else {
          // Observed surface mesh
          const origMat = child.material as any;
          const origMap = origMat?.map || (Array.isArray(origMat) ? origMat[0]?.map : null);
          const hasColors = Boolean(child.geometry?.attributes?.color);

          if (origMap) {
            child.userData.origMap = origMap;
            origMap.flipY = false;
            origMap.needsUpdate = true;
            child.material = new THREE.MeshStandardMaterial({
              map: origMap,
              roughness: 0.85,
              metalness: 0.05,
              side: THREE.DoubleSide,
              polygonOffset: true,
              polygonOffsetFactor: 2.0,
              polygonOffsetUnits: 2.0,
            });
          } else {
            child.userData.origMap = null;
            // Untextured: use authentic vertex colors if present, otherwise clean photogrammetry neutral
            child.material = new THREE.MeshStandardMaterial({
              vertexColors: hasColors,
              color: hasColors ? 0xffffff : 0xd8dbe2,
              roughness: 0.70,
              metalness: 0.05,
              side: THREE.DoubleSide,
              flatShading: false,
              polygonOffset: true,
              polygonOffsetFactor: 2.0,
              polygonOffsetUnits: 2.0,
            });
          }
        }
      }
    });

    const b = new THREE.Box3();
    let foundTight = false;
    group.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        const name = (child.name || '').toLowerCase();
        if (!name.includes('base') && !name.includes('inferred')) {
          b.expandByObject(child);
          foundTight = true;
        }
      }
    });
    if (!foundTight) {
      group.traverse((child) => {
        if (child instanceof THREE.Points) {
          const name = (child.name || '').toLowerCase();
          if (name.includes('dense')) {
            b.expandByObject(child);
            foundTight = true;
          }
        }
      });
    }
    if (!foundTight) {
      b.setFromObject(group);
    }
    const center = b.getCenter(new THREE.Vector3());
    group.position.set(-center.x, -center.y, -center.z);

    return { modelGroup: group };
  }, [scene]);

  // Whenever pitch, roll, or modelGroup changes, update world transform and notify onFramed
  useEffect(() => {
    if (pivotRef.current) {
      pivotRef.current.updateMatrixWorld(true);
      const worldBox = new THREE.Box3().setFromObject(pivotRef.current);
      if (!worldBox.isEmpty()) {
        onFramed?.(worldBox);
      }
    }
  }, [pitch, roll, modelGroup, onFramed]);

  // Dynamically update layer visibility, point sizing, and hybrid rendering in-place
  useEffect(() => {
    modelGroup.traverse((child) => {
      if (child instanceof THREE.Points) {
        const name = (child.name || '').toLowerCase();
        const count = child.geometry?.attributes?.position?.count || 0;
        const isSparse = name.includes('sparse') || (count > 0 && count < 50000 && !name.includes('dense'));
        const isDense = !isSparse;

        if (viewMode === 'SPARSE') {
          child.visible = isSparse;
          if (child.material instanceof THREE.PointsMaterial) {
            child.material.size = Math.max(pointSize * 3.5, 0.08);
            child.material.color.set(0xffaa22); // Golden amber for SfM feature constellation
            child.material.vertexColors = false;
            child.material.needsUpdate = true;
          }
        } else if (viewMode === 'DENSE') {
          child.visible = isDense;
          if (child.material instanceof THREE.PointsMaterial) {
            child.material.size = pointSize;
            child.material.color.set(0xffffff);
            child.material.vertexColors = true;
            child.material.needsUpdate = true;
          }
        } else if (viewMode === 'HYBRID' || viewMode === 'PRESENTATION') {
          child.visible = isDense;
          if (child.material instanceof THREE.PointsMaterial) {
            child.material.size = Math.max(0.008, pointSize * 0.6);
            child.material.color.set(0xffffff);
            child.material.vertexColors = true;
            child.material.needsUpdate = true;
          }
        } else {
          // OBSERVED, MESH, TEXTURED, INFERRED: hide dense points
          child.visible = false;
        }
      } else if (child instanceof THREE.Mesh) {
        const name = (child.name || '').toLowerCase();
        const isBase = name.includes('presentation_base');
        const isInferred = name.includes('inferred') || name.includes('completion');
        const isObserved = !isBase && !isInferred;

        if (isBase) {
          child.visible = showBase || viewMode === 'PRESENTATION';
        } else if (isInferred) {
          if (viewMode === 'INFERRED') {
            child.visible = true;
          } else if (viewMode === 'OBSERVED' || viewMode === 'DENSE' || viewMode === 'SPARSE') {
            child.visible = false;
          } else {
            // MESH, HYBRID, PRESENTATION, TEXTURED
            child.visible = showInferred;
          }
        } else if (isObserved) {
          if (viewMode === 'INFERRED' || viewMode === 'DENSE' || viewMode === 'SPARSE') {
            child.visible = false;
          } else {
            // OBSERVED, MESH, HYBRID, PRESENTATION, TEXTURED
            child.visible = true;
          }
        }

        if (isObserved && child.material instanceof THREE.MeshStandardMaterial) {
          const origMap = child.userData.origMap;
          const hasColors = Boolean(child.geometry?.attributes?.color);
          if (viewMode === 'MESH') {
            // MESH: Geometric surface visualization without texture map overlay
            child.material.map = null;
            child.material.vertexColors = hasColors;
            child.material.color.set(hasColors ? 0xffffff : 0xd2d7e2);
            child.material.roughness = 0.65;
            child.material.metalness = 0.05;
          } else {
            // TEXTURED, HYBRID, OBSERVED, PRESENTATION: Photorealistic texture map
            if (origMap) {
              child.material.map = origMap;
              child.material.vertexColors = false;
              child.material.color.set(0xffffff);
              child.material.roughness = 0.85;
              child.material.metalness = 0.05;
            } else {
              child.material.map = null;
              child.material.vertexColors = hasColors;
              child.material.color.set(hasColors ? 0xffffff : 0xd8dbe2);
            }
          }
          child.material.needsUpdate = true;
        }

        if (child.material) {
          const mats = Array.isArray(child.material) ? child.material : [child.material];
          mats.forEach((m) => {
            if (!isBase) {
              m.wireframe = wireframe;
            }
            m.needsUpdate = true;
          });
        }
      }
    });
  }, [modelGroup, viewMode, pointSize, wireframe, showBase, showInferred]);

  return (
    <group
      ref={pivotRef}
      rotation={[
        THREE.MathUtils.degToRad(pitch),
        0,
        THREE.MathUtils.degToRad(roll),
      ]}
    >
      <primitive object={modelGroup} />
    </group>
  );
}

useGLTF.preload;
