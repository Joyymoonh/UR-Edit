export type JobStatus =
  | 'queued'
  | 'extracting'
  | 'masking'
  | 'propagating'
  | 'editing'
  | 'smoothing'
  | 'saving'
  | 'finished'
  | 'failed';

export interface VideoEditStatus {
  jobId: string;
  status: JobStatus;
  progress: number;
  currentFrame: number | null;
  totalFrames: number | null;
  message: string | null;
  error: string | null;
  outputVideoPath: string | null;
  maskPreviewPath: string | null;
  logsPath: string | null;
}

export type EditMode = 'video' | 'single_frame';

export interface VideoEditRequest {
  jobId: string;
  inputVideoPath: string;
  outputDir: string;
  maskPrompt: string;
  editPrompt: string;
  mode: EditMode;
  maxFrames: number | null;
  imageGuidanceScale: number;
  guidanceScale: number;
  erodeKernel: number;
  numInferenceSteps: number;
  applySmoothing: boolean;
  smoothingWindow: number;
  smoothingAlpha: number;
  extra?: Record<string, unknown>;
}
