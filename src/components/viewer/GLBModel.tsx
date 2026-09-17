import { useEffect, useMemo } from 'react';
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

// Fast Jacobi eigenvalue solver for 3x3 symmetric covariance matrix to find dominant ground plane
function computeSymmetricEigen3x3(A: number[][]): { normal: [number, number, number] } {
  const V = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ];
  const a = [
    [A[0][0], A[0][1], A[0][2]],
    [A[0][1], A[1][1], A[1][2]],
    [A[0][2], A[1][2], A[2][2]],
  ];

  for (let iter = 0; iter < 50; iter++) {
    let p = 0, q = 1;
    let maxOff = Math.abs(a[0][1]);
    if (Math.abs(a[0][2]) > maxOff) { maxOff = Math.abs(a[0][2]); p = 0; q = 2; }
    if (Math.abs(a[1][2]) > maxOff) { maxOff = Math.abs(a[1][2]); p = 1; q = 2; }

    if (maxOff < 1e-8) break;

    const app = a[p][p], aqq = a[q][q], apq = a[p][q];
    const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
    const c = Math.cos(phi), s = Math.sin(phi);

    for (let i = 0; i < 3; i++) {
      if (i !== p && i !== q) {
        const a_ip = a[i][p];
        const a_iq = a[i][q];
        a[i][p] = a[p][i] = c * a_ip - s * a_iq;
        a[i][q] = a[q][i] = s * a_ip + c * a_iq;
      }
      const v_ip = V[i][p];
      const v_iq = V[i][q];
      V[i][p] = c * v_ip - s * v_iq;
      V[i][q] = s * v_ip + c * v_iq;
    }
    a[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    a[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    a[p][q] = a[q][p] = 0;
  }

  const eigenvalues = [a[0][0], a[1][1], a[2][2]];
  const indices = [0, 1, 2].sort((i, j) => eigenvalues[i] - eigenvalues[j]);
  const minIdx = indices[0];
  return { normal: [V[0][minIdx], V[1][minIdx], V[2][minIdx]] };
}

export function GLBModel({
  url,
  viewMode = 'HYBRID',
  pointSize = 0.022,
  wireframe = false,
  showBase = false,
  showInferred = true,
  onFramed,
}: Props) {
  const { scene } = useGLTF(url);

  // Initialize scene and configure photogrammetric materials once per model
  const { modelGroup, box } = useMemo(() => {
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

    // Auto-Leveling / Ground-Plane Alignment:
    // Sample vertices from observed meshes (or dense points) to detect the dominant ground plane normal
    const samplePts: number[] = [];
    group.traverse((child) => {
      if (child instanceof THREE.Mesh || child instanceof THREE.Points) {
        const name = (child.name || '').toLowerCase();
        if (!name.includes('base') && !name.includes('inferred')) {
          const pos = child.geometry?.attributes?.position;
          if (pos && pos.count > 0) {
            const step = Math.max(1, Math.floor(pos.count / 1500));
            for (let i = 0; i < pos.count; i += step) {
              samplePts.push(pos.getX(i), pos.getY(i), pos.getZ(i));
            }
          }
        }
      }
    });

    if (samplePts.length >= 300) {
      const nPts = samplePts.length / 3;
      let meanX = 0, meanY = 0, meanZ = 0;
      for (let i = 0; i < samplePts.length; i += 3) {
        meanX += samplePts[i];
        meanY += samplePts[i + 1];
        meanZ += samplePts[i + 2];
      }
      meanX /= nPts;
      meanY /= nPts;
      meanZ /= nPts;

      let cxx = 0, cyy = 0, czz = 0, cxy = 0, cxz = 0, cyz = 0;
      for (let i = 0; i < samplePts.length; i += 3) {
        const dx = samplePts[i] - meanX;
        const dy = samplePts[i + 1] - meanY;
        const dz = samplePts[i + 2] - meanZ;
        cxx += dx * dx; cyy += dy * dy; czz += dz * dz;
        cxy += dx * dy; cxz += dx * dz; cyz += dy * dz;
      }
      cxx /= nPts; cyy /= nPts; czz /= nPts;
      cxy /= nPts; cxz /= nPts; cyz /= nPts;

      const cov = [
        [cxx, cxy, cxz],
        [cxy, cyy, cyz],
        [cxz, cyz, czz],
      ];
      const { normal } = computeSymmetricEigen3x3(cov);
      const groundNormal = new THREE.Vector3(normal[0], normal[1], normal[2]);
      if (groundNormal.y < 0) groundNormal.negate();

      const vertical = new THREE.Vector3(0, 1, 0);
      const angle = groundNormal.angleTo(vertical);
      // If dominant normal is reasonably vertical (> 30 deg above horizontal) and tilted by > 1.5 deg (> 0.026 rad)
      if (groundNormal.y > 0.5 && angle > 0.026) {
        const alignQuat = new THREE.Quaternion().setFromUnitVectors(groundNormal, vertical);
        group.traverse((child) => {
          if (child instanceof THREE.Mesh || child instanceof THREE.Points) {
            child.geometry = child.geometry.clone();
            child.geometry.applyQuaternion(alignQuat);
            if (child instanceof THREE.Mesh) {
              try {
                child.geometry.computeVertexNormals();
              } catch {
                /* ignore */
              }
            }
          }
        });
      }
    }

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
    group.position.sub(center);

    const centeredBox = new THREE.Box3().setFromCenterAndSize(
      new THREE.Vector3(0, 0, 0),
      b.getSize(new THREE.Vector3())
    );

    return { modelGroup: group, box: centeredBox };
  }, [scene]);

  // Frame camera once when a new model is loaded
  useEffect(() => {
    if (box && !box.isEmpty()) {
      onFramed?.(box);
    }
  }, [box, onFramed]);

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

  return <primitive object={modelGroup} />;
}

useGLTF.preload;
