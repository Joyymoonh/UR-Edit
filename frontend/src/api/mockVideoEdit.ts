import type { JobStatus, VideoEditRequest, VideoEditStatus } from '../types';

interface InternalJob {
  request: VideoEditRequest;
  status: VideoEditStatus;
  stageIndex: number;
  timer?: ReturnType<typeof setInterval>;
  startedAt: number;
}

const STAGES: JobStatus[] = ['queued', 'extracting', 'masking', 'propagating', 'editing', 'smoothing', 'saving', 'finished'];
const STAGE_THRESHOLDS = [0, 0.08, 0.15, 0.35, 0.6, 0.8, 0.93, 1];
const STAGE_MESSAGE: Partial<Record<JobStatus, string>> = {
  queued: '等待 GPU 空闲...',
  extracting: '正在提取帧 / Resize',
  masking: 'SAM3 正在生成第一帧掩膜',
  propagating: '基于光流传播掩膜',
  editing: 'Instruct-Pix2Pix 渲染中...',
  smoothing: '应用时序平滑',
  saving: '写入视频/可视化',
  finished: '完成，等待下载',
};

const jobStore = new Map<string, InternalJob>();

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

function randomBetween(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

function initStatus(request: VideoEditRequest): VideoEditStatus {
  return {
    jobId: request.jobId,
    status: 'queued',
    progress: 0,
    currentFrame: 0,
    totalFrames: request.maxFrames,
    message: STAGE_MESSAGE.queued ?? null,
    error: null,
    outputVideoPath: null,
    maskPreviewPath: null,
    logsPath: `${request.outputDir}/run.log`,
  };
}

function advanceJob(jobId: string) {
  const job = jobStore.get(jobId);
  if (!job || job.status.status === 'finished' || job.status.status === 'failed') {
    if (job?.timer) clearInterval(job.timer);
    return;
  }

  const delta = randomBetween(0.05, 0.2);
  const nextProgress = Math.min(job.status.progress + delta, 1);
  job.status.progress = nextProgress;

  if (job.status.totalFrames) {
    job.status.currentFrame = Math.min(job.status.totalFrames, Math.round(job.status.totalFrames * nextProgress));
  }

  const inferredStage = STAGE_THRESHOLDS.reduce((acc, threshold, index) => {
    if (nextProgress >= threshold) {
      return index;
    }
    return acc;
  }, job.stageIndex);
  const nextStageIndex = Math.min(STAGES.length - 1, Math.max(job.stageIndex, inferredStage));

  if (nextStageIndex > job.stageIndex) {
    job.stageIndex = nextStageIndex;
    job.status.status = STAGES[nextStageIndex];
    job.status.message = STAGE_MESSAGE[job.status.status] ?? null;
  }

  if (Math.random() < 0.02 && job.status.status !== 'finished' && job.status.status !== 'failed') {
    job.status.status = 'failed';
    job.status.error = '推理失败：显存不足 (模拟)';
    job.status.message = null;
    if (job.timer) clearInterval(job.timer);
    return;
  }

  if (nextProgress >= 1 && job.status.status === 'finished') {
    job.status.currentFrame = job.status.totalFrames;
    job.status.outputVideoPath = `${job.request.outputDir}/result.mp4`;
    job.status.maskPreviewPath = `${job.request.outputDir}/first_mask.png`;
    job.status.message = STAGE_MESSAGE.finished ?? '任务完成';
    if (job.timer) clearInterval(job.timer);
  }
}

function scheduleJob(job: InternalJob) {
  job.timer = setInterval(() => advanceJob(job.status.jobId), randomBetween(1800, 2600));
}

export async function runVideoEdit(request: VideoEditRequest): Promise<{ jobId: string }> {
  await delay(randomBetween(150, 350));
  const status = initStatus(request);
  const job: InternalJob = {
    request,
    status,
    stageIndex: 0,
    startedAt: Date.now(),
  };
  jobStore.set(request.jobId, job);
  scheduleJob(job);
  return { jobId: request.jobId };
}

export async function getVideoEditStatus(jobId: string): Promise<VideoEditStatus> {
  await delay(randomBetween(80, 200));
  const job = jobStore.get(jobId);
  if (!job) {
    throw new Error(`Job ${jobId} not found`);
  }
  return { ...job.status };
}

export async function bootstrapMockJobs(requests: VideoEditRequest[]): Promise<
  Array<{
    request: VideoEditRequest;
    status: VideoEditStatus;
  }>
> {
  const results: Array<{ request: VideoEditRequest; status: VideoEditStatus }> = [];
  for (const request of requests) {
    await runVideoEdit(request);
    const status = await getVideoEditStatus(request.jobId);
    results.push({ request, status });
  }
  return results;
}
