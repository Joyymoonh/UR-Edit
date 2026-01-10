import { useEffect, useMemo, useRef, useState } from 'react';
import JobForm, { type JobFormValues } from './components/JobForm';
import JobCard, { type JobMeta } from './components/JobCard';
import ResultPreview, { type PreviewVariant } from './components/ResultPreview';
import { bootstrapMockJobs, getVideoEditStatus, runVideoEdit } from './api/mockVideoEdit';
import { mockUpload } from './api/mockUpload';
import type { VideoEditRequest, VideoEditStatus } from './types';

const SAMPLE_REQUESTS: VideoEditRequest[] = [
  {
    jobId: 'JOB-24001',
    inputVideoPath: '/examples/lookbook_scene_12.mp4',
    outputDir: '/outputs/JOB-24001',
    maskPrompt: 'jacket',
    editPrompt: 'turn the jacket into a neon cyberpunk style armor',
    mode: 'video',
    maxFrames: 80,
    imageGuidanceScale: 1.4,
    guidanceScale: 8.8,
    erodeKernel: 12,
    numInferenceSteps: 45,
    applySmoothing: true,
    smoothingWindow: 5,
    smoothingAlpha: 0.35,
    extra: { jobName: 'Neon Jacket v2' },
  },
  {
    jobId: 'JOB-24002',
    inputVideoPath: '/examples/hair_color_pass.mp4',
    outputDir: '/outputs/JOB-24002',
    maskPrompt: 'long hair',
    editPrompt: 'change the hair to silky silver with glitter',
    mode: 'video',
    maxFrames: 120,
    imageGuidanceScale: 1.5,
    guidanceScale: 9.2,
    erodeKernel: 10,
    numInferenceSteps: 50,
    applySmoothing: true,
    smoothingWindow: 7,
    smoothingAlpha: 0.28,
    extra: { jobName: 'Hair Color Pass' },
  },
  {
    jobId: 'JOB-23990',
    inputVideoPath: '/examples/product_shot.mp4',
    outputDir: '/outputs/JOB-23990',
    maskPrompt: 'background clutter',
    editPrompt: 'replace clutter with a clean concrete wall and soft shadows',
    mode: 'video',
    maxFrames: 60,
    imageGuidanceScale: 1.2,
    guidanceScale: 7.8,
    erodeKernel: 8,
    numInferenceSteps: 40,
    applySmoothing: false,
    smoothingWindow: 3,
    smoothingAlpha: 0.2,
    extra: { jobName: 'Product Shot Cleanup' },
  },
];

const buildJobMeta = (request: VideoEditRequest, status: VideoEditStatus, previous?: JobMeta): JobMeta => {
  const startedAt = previous?.startedAt ?? Date.now();
  const elapsedMinutes = Math.max((Date.now() - startedAt) / 60000, 0.1);
  const durationLabel =
    status.status === 'finished'
      ? `用时 ${elapsedMinutes.toFixed(1)}m`
      : status.status === 'queued'
        ? '等待执行'
        : `已运行 ${elapsedMinutes.toFixed(1)}m`;

  return {
    ...status,
    request,
    name: (request.extra?.jobName as string) || previous?.name || request.jobId,
    maskPrompt: previous?.maskPrompt ?? request.maskPrompt,
    editPrompt: previous?.editPrompt ?? request.editPrompt,
    updatedAt: new Date().toLocaleTimeString(),
    duration: durationLabel,
    startedAt,
    logsPath: status.logsPath ?? `${request.outputDir}/run.log`,
    uploadPreviewUrl: previous?.uploadPreviewUrl ?? null,
  };
};

const buildRequestFromForm = (values: JobFormValues, jobId: string, inputVideoPath: string): VideoEditRequest => ({
  jobId,
  inputVideoPath,
  outputDir: `/outputs/${jobId}`,
  maskPrompt: values.maskPrompt,
  editPrompt: values.editPrompt,
  mode: 'video',
  maxFrames: values.maxFrames,
  imageGuidanceScale: values.imageGuidanceScale,
  guidanceScale: values.guidanceScale,
  erodeKernel: values.erodeKernel,
  numInferenceSteps: 50,
  applySmoothing: true,
  smoothingWindow: 5,
  smoothingAlpha: 0.3,
  extra: {
    jobName: values.jobName || jobId,
    notes: values.notes,
  },
});

function App() {
  const [jobs, setJobs] = useState<JobMeta[]>([]);
  const [initialised, setInitialised] = useState(false);
  const [previewContext, setPreviewContext] = useState<{ job: JobMeta; variant: PreviewVariant } | null>(null);
  const jobIdsKey = jobs.map((job) => job.jobId).join('|');
  const jobsRef = useRef<JobMeta[]>([]);

  useEffect(() => {
    jobsRef.current = jobs;
  }, [jobs]);

  useEffect(() => {
    if (initialised) {
      return;
    }
    (async () => {
      const seeded = await bootstrapMockJobs(SAMPLE_REQUESTS);
      const metas = seeded.map(({ request, status }) => {
        const placeholder: JobMeta = {
          ...status,
          request,
          name: (request.extra?.jobName as string) || request.jobId,
          maskPrompt: request.maskPrompt,
          editPrompt: request.editPrompt,
          updatedAt: new Date().toLocaleTimeString(),
          duration: '排队中',
          startedAt: Date.now() - Math.floor(Math.random() * 4 * 60 * 1000),
          uploadPreviewUrl: null,
        };
        return buildJobMeta(request, status, placeholder);
      });
      setJobs(metas);
      setInitialised(true);
    })();
  }, [initialised]);

  useEffect(() => {
    if (!jobs.length) {
      return;
    }
    let active = true;
    const poll = async () => {
      const updates = await Promise.all(
        jobsRef.current.map((job) =>
          getVideoEditStatus(job.jobId)
            .then((status) => ({ status, job }))
            .catch(() => null),
        ),
      );
      if (!active) return;
      setJobs((prev) =>
        prev.map((job) => {
          const next = updates.find((update) => update?.job.jobId === job.jobId);
          if (!next?.status) {
            return job;
          }
          return buildJobMeta(job.request, next.status, job);
        }),
      );
    };
    const timer = setInterval(poll, 2500);
    poll();
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [jobIdsKey]);

  const runningJobs = useMemo(() => jobs.filter((job) => job.status !== 'finished' && job.status !== 'failed'), [jobs]);

  const handleCreateJob = async (values: JobFormValues) => {
    const jobId = `JOB-${Date.now().toString().slice(-5)}`;
    let uploadPreviewUrl: string | null = null;
    let inputPath = `/uploads/${values.inputVideoName || jobId}.mp4`;

    if (values.file) {
      uploadPreviewUrl = URL.createObjectURL(values.file);
      const { serverPath } = await mockUpload(values.file, jobId);
      inputPath = serverPath;
    }

    const request = buildRequestFromForm(values, jobId, inputPath);

    try {
      await runVideoEdit(request);
      const status = await getVideoEditStatus(jobId);
      const placeholder: JobMeta = {
        ...status,
        request,
        name: request.extra?.jobName as string,
        maskPrompt: request.maskPrompt,
        editPrompt: request.editPrompt,
        updatedAt: new Date().toLocaleTimeString(),
        duration: '等待执行',
        startedAt: Date.now(),
        uploadPreviewUrl,
      };
      const meta = buildJobMeta(request, status, placeholder);
      setJobs((prev) => [meta, ...prev]);
    } catch (error) {
      if (uploadPreviewUrl) {
        URL.revokeObjectURL(uploadPreviewUrl);
      }
      console.error('Failed to create job', error);
    }
  };

  const handlePreview = (job: JobMeta, variant: PreviewVariant) => {
    setPreviewContext({ job, variant });
  };

  return (
    <div className="page-shell">
      <header className="hero">
        <div>
          <p className="hero-subtitle">UR-Edit 控制台</p>
          <h1>分布式视频编辑面板</h1>
          <p className="hero-desc">
            提交新任务、追踪 GPU 状态、快速预览 Mask 结果。当前 {runningJobs.length} 个作业正在运行（模拟接口模式）。
          </p>
        </div>
        <div className="hero-highlights">
          <div>
            <p className="hero-metric-label">平均耗时</p>
            <p className="hero-metric-value">3.4 分钟</p>
          </div>
          <div>
            <p className="hero-metric-label">成功率</p>
            <p className="hero-metric-value">97%</p>
          </div>
          <div>
            <p className="hero-metric-label">GPU 占用</p>
            <p className="hero-metric-value">72%</p>
          </div>
        </div>
      </header>

      <section className="panel-grid">
        <div className="panel panel-form">
          <h2>创建新任务</h2>
          <p className="panel-tip">表单提交后将通过 run_video_edit 接口推送到 GPU 队列（当前为模拟数据）。</p>
          <JobForm onSubmit={handleCreateJob} />
        </div>
        <div className="panel panel-jobs">
          <div className="panel-header">
            <h2>任务队列</h2>
            <span className="pill">接口模式（Mock）</span>
          </div>
          <div className="jobs-list">
            {jobs.map((job) => (
              <JobCard key={job.jobId} job={job} onPreview={handlePreview} />
            ))}
          </div>
        </div>
      </section>

      <ResultPreview context={previewContext} onClose={() => setPreviewContext(null)} />
    </div>
  );
}

export default App;
