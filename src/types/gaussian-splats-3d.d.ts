declare module '@mkkellogg/gaussian-splats-3d' {
  export const SceneFormat: {
    Ply: number;
    Splat: number;
    KSplat: number;
    Spz: number;
  };
  export class Viewer {
    constructor(options?: any);
    addSplatScene(url: string, options?: any): Promise<any>;
    start(): void;
    stop(): void;
    dispose(): void;
  }
  export class DropInViewer {
    constructor(options?: any);
    addSplatScene(url: string, options?: any): Promise<any>;
  }
}
