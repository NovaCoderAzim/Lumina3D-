declare module '@mkkellogg/gaussian-splats-3d' {
  export const SceneFormat: {
    Ply: number;
    Splat: number;
    KSplat: number;
    Spz: number;
  };
  export const RenderMode: {
    Always: number;
    OnChange: number;
    Never: number;
  };
  export class Viewer {
    constructor(options?: any);
    addSplatScene(url: string, options?: any): Promise<any>;
    start(): void;
    stop(): void;
    dispose(): Promise<any>;
    forceRenderNextFrame(): void;
    threeScene: any;
    camera: any;
    renderer: any;
    controls: any;
    splatMesh: any;
  }
  export class DropInViewer {
    constructor(options?: any);
    addSplatScene(url: string, options?: any): Promise<any>;
    dispose(): Promise<any>;
  }
}
