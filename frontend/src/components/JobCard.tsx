import type { VideoEditRequest, VideoEditStatus } from '../types';

export interface JobMeta extends VideoEditStatus {
  request: VideoEditRequest;
  name: string;
  maskPrompt: string;
  editPrompt: string;
  updatedAt: string;
  duration: string;
  startedAt: number;
  uploadPreviewUrl?: string | null;
}

const statusLabel: Record<VideoEditStatus['status'], string> = {
  queued: '排队中',
  extracting: '提帧',
  masking: '分割',
  propagating: '掩膜传播',
  editing: '渲染中',
  smoothing: '平滑',
  saving: '写入',
  finished: '完成',
  failed: '失败',
};

const statusTone: Record<VideoEditStatus['status'], string> = {
  queued: 'neutral',
  extracting: 'neutral',
  masking: 'neutral',
  propagating: 'neutral',
  editing: 'blue',
  smoothing: 'blue',
  saving: 'blue',
  finished: 'green',
  failed: 'red',
};

interface JobCardProps {
  job: JobMeta;
  onPreview: (job: JobMeta, variant: 'output' | 'mask' | 'input') => void;
}

function JobCard({ job, onPreview }: JobCardProps) {
  const tone = statusTone[job.status];
  const formatProgress = Math.round(job.progress * 100);

  const canPreviewOutput = Boolean(job.outputVideoPath || job.uploadPreviewUrl);
  const canPreviewMask = Boolean(job.maskPreviewPath);
  const canPreviewInput = Boolean(job.uploadPreviewUrl);

  return (
    <article className={`job-card job-card--${tone}`}>
      <header className="job-card__header">
        <div>
          <p className="job-id">{job.jobId}</p>
          <h3>{job.name}</h3>
        </div>
        <div className={`status-badge status-${tone}`}>{statusLabel[job.status]}</div>
      </header>

      <div className="progress-wrapper">
        <div className="progress-bar">
          <span style={{ width: `${formatProgress}%` }} />
        </div>
        <p className="progress-label">{formatProgress}%</p>
      </div>

      <section className="prompt-block">
        <p>
          <strong>Mask Prompt</strong>
          <span>{job.maskPrompt}</span>
        </p>
        <p>
          <strong>Edit Prompt</strong>
          <span>{job.editPrompt}</span>
        </p>
      </section>

      <section className="job-meta">
        <span>{job.updatedAt}</span>
        <span>{job.duration}</span>
      </section>

      {job.error ? (
        <p className="job-error">{job.error}</p>
      ) : job.message ? (
        <p className="job-message">{job.message}</p>
      ) : null}

      <footer className="job-actions">
        <button type="button" disabled={!canPreviewInput} onClick={() => onPreview(job, 'input')}>
          预览原片
        </button>
        <button type="button" disabled={!canPreviewOutput} onClick={() => onPreview(job, 'output')}>
          预览结果
        </button>
        <button type="button" disabled={!canPreviewMask} onClick={() => onPreview(job, 'mask')}>
          查看 Mask
        </button>
        <a href={job.logsPath ?? '#'} target="_blank" rel="noreferrer">
          查看日志
        </a>
      </footer>
    </article>
  );
}

export default JobCard;
