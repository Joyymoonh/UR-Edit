import type { JobMeta } from './JobCard';

type PreviewVariant = 'output' | 'mask' | 'input';

interface ResultPreviewProps {
  context: { job: JobMeta; variant: PreviewVariant } | null;
  onClose: () => void;
}

const variantTitle: Record<PreviewVariant, string> = {
  input: '原始视频',
  output: '生成结果',
  mask: 'Mask 预览',
};

function resolveSource(job: JobMeta, variant: PreviewVariant): string | null {
  if (variant === 'input') {
    return job.uploadPreviewUrl ?? null;
  }
  if (variant === 'output') {
    return job.outputVideoPath ?? job.uploadPreviewUrl ?? null;
  }
  return job.maskPreviewPath ?? null;
}

function ResultPreview({ context, onClose }: ResultPreviewProps) {
  if (!context) return null;

  const { job, variant } = context;
  const source = resolveSource(job, variant);

  return (
    <div className="preview-backdrop" onClick={onClose}>
      <div className="preview-card" onClick={(e) => e.stopPropagation()}>
        <header>
          <div>
            <p className="preview-label">{job.jobId}</p>
            <h4>
              {job.name} · {variantTitle[variant]}
            </h4>
          </div>
          <button type="button" onClick={onClose}>
            ×
          </button>
        </header>
        {source ? (
          <video key={source} src={source} controls playsInline preload="metadata" />
        ) : (
          <p className="preview-empty">暂无可预览的文件</p>
        )}
      </div>
    </div>
  );
}

export type { PreviewVariant };
export default ResultPreview;
