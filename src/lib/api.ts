import type { AnalyticsResponse, ProjectStatusResponse, ProjectSummary, SemanticObject } from '../types';

// Base URL is intentionally configurable — components never hardcode routes.
const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init);
  if (!res.ok) {
    throw new Error(`API ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export interface ModelInfoResponse {
  project_id: string;
  model_url: string | null; // backend-relative, e.g. /api/projects/x/model/file
  format: string;
  available: boolean;
}

export const api = {
  listProjects() {
    return request<{ projects: ProjectSummary[] }>('/projects');
  },
  createProject(name: string) {
    return request<{ project_id: string }>('/projects', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
  },
  uploadVideo(projectId: string, file: File, onProgress?: (pct: number) => void) {
    // Uses XHR instead of fetch so real upload progress can drive the UI.
    return new Promise<void>((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', `${BASE_URL}/projects/${projectId}/upload`);
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress((e.loaded / e.total) * 100);
      };
      xhr.onload = () => (xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error(xhr.statusText)));
      xhr.onerror = () => reject(new Error('Upload failed'));
      const form = new FormData();
      form.append('video', file);
      xhr.send(form);
    });
  },
  getStatus(projectId: string) {
    return request<ProjectStatusResponse>(`/projects/${projectId}/status`);
  },
  // Returns model metadata; model_url points at the binary .glb endpoint.
  getModelInfo(projectId: string) {
    return request<ModelInfoResponse>(`/projects/${projectId}/model`);
  },
  // Absolute URL the Three.js loader can fetch directly.
  modelFileUrl(projectId: string) {
    return `${BASE_URL}/projects/${projectId}/model/file`;
  },
  getSemantic(projectId: string) {
    return request<SemanticObject[]>(`/projects/${projectId}/semantic`);
  },
  getAnalytics(projectId: string) {
    return request<AnalyticsResponse>(`/projects/${projectId}/analytics`);
  },
  getCapture(projectId: string) {
    return request<import('../types').CaptureQuality>(`/projects/${projectId}/capture`);
  },
};
