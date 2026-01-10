import { useEffect, useMemo, useState } from 'react';
import type { ChangeEvent, FormEvent } from 'react';

export interface JobFormValues {
  jobName: string;
  inputVideoName: string;
  maskPrompt: string;
  editPrompt: string;
  imageGuidanceScale: number;
  guidanceScale: number;
  erodeKernel: number;
  maxFrames: number;
  notes?: string;
  file?: File;
}

interface JobFormProps {
  onSubmit: (values: JobFormValues) => void;
}

const suggestions: Array<Pick<JobFormValues, 'maskPrompt' | 'editPrompt'>> = [
  { maskPrompt: 'dress', editPrompt: 'turn the dress into a sleek black velvet gown' },
  { maskPrompt: 'sky', editPrompt: 'replace sky with sunset gradient and neon clouds' },
  { maskPrompt: 'hair', editPrompt: 'change to silver bob cut with metallic sheen' },
];

const defaultValues: JobFormValues = {
  jobName: '',
  inputVideoName: '',
  maskPrompt: '',
  editPrompt: '',
  imageGuidanceScale: 1.4,
  guidanceScale: 8.5,
  erodeKernel: 10,
  maxFrames: 90,
  notes: '',
};

function JobForm({ onSubmit }: JobFormProps) {
  const [values, setValues] = useState<JobFormValues>(defaultValues);
  const [fileLabel, setFileLabel] = useState('?????');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [filePreview, setFilePreview] = useState<string | null>(null);

  const isValid = useMemo(() => !!values.maskPrompt && !!values.editPrompt && !!selectedFile, [values, selectedFile]);

  const handleChange = <K extends keyof JobFormValues>(field: K, val: JobFormValues[K]) => {
    setValues((prev) => ({ ...prev, [field]: val }));
  };

  const handleFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setFileLabel(file.name);
      handleChange('inputVideoName', file.name);
      setSelectedFile(file);
      if (filePreview) {
        URL.revokeObjectURL(filePreview);
      }
      setFilePreview(URL.createObjectURL(file));
    }
  };

  const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!isValid) return;
    onSubmit({ ...values, file: selectedFile ?? undefined });
    setValues(defaultValues);
    setFileLabel('?????');
    setSelectedFile(null);
    if (filePreview) {
      URL.revokeObjectURL(filePreview);
      setFilePreview(null);
    }
  };

  useEffect(() => {
    return () => {
      if (filePreview) {
        URL.revokeObjectURL(filePreview);
      }
    };
  }, [filePreview]);

  const applySuggestion = (index: number) => {
    const suggestion = suggestions[index];
    setValues((prev) => ({ ...prev, ...suggestion }));
  };

  return (
    <form className="job-form" onSubmit={handleSubmit}>
      <label className="form-field">
        <span>????</span>
        <input
          type="text"
          placeholder="???Lookbook-Scene-12"
          value={values.jobName}
          onChange={(e) => handleChange('jobName', e.target.value)}
        />
      </label>

      <label className="form-field file-field">
        <span>????</span>
        <div className="file-input">
          <input type="file" id="video-input" accept="video/mp4,video/quicktime" onChange={handleFile} />
          <label htmlFor="video-input">????</label>
          <p>{fileLabel}</p>
        </div>
      </label>
      {filePreview ? (
        <div className="upload-preview">
          <video src={filePreview} controls preload="metadata" />
          <small>??????????????</small>
        </div>
      ) : null}

      <label className="form-field">
        <span>Mask Prompt</span>
        <div className="input-with-actions">
          <input
            type="text"
            placeholder="??????????"
            value={values.maskPrompt}
            onChange={(e) => handleChange('maskPrompt', e.target.value)}
          />
          <div className="suggestions">
            {suggestions.map((s, index) => (
              <button type="button" key={s.maskPrompt} onClick={() => applySuggestion(index)}>
                {s.maskPrompt}
              </button>
            ))}
          </div>
        </div>
      </label>

      <label className="form-field">
        <span>Edit Prompt</span>
        <textarea
          rows={3}
          placeholder="???????"
          value={values.editPrompt}
          onChange={(e) => handleChange('editPrompt', e.target.value)}
        />
      </label>

      <div className="form-row">
        <label className="form-field">
          <span>Image Guidance</span>
          <input
            type="range"
            min="0.8"
            max="2.0"
            step="0.1"
            value={values.imageGuidanceScale}
            onChange={(e) => handleChange('imageGuidanceScale', Number(e.target.value))}
          />
          <small>{values.imageGuidanceScale.toFixed(1)}</small>
        </label>

        <label className="form-field">
          <span>Text Guidance</span>
          <input
            type="range"
            min="6"
            max="12"
            step="0.1"
            value={values.guidanceScale}
            onChange={(e) => handleChange('guidanceScale', Number(e.target.value))}
          />
          <small>{values.guidanceScale.toFixed(1)}</small>
        </label>
      </div>

      <div className="form-row">
        <label className="form-field">
          <span>Erode Kernel</span>
          <input
            type="number"
            min={0}
            max={30}
            value={values.erodeKernel}
            onChange={(e) => handleChange('erodeKernel', Number(e.target.value))}
          />
        </label>

        <label className="form-field">
          <span>????</span>
          <input
            type="number"
            min={30}
            max={300}
            step={10}
            value={values.maxFrames}
            onChange={(e) => handleChange('maxFrames', Number(e.target.value))}
          />
        </label>
      </div>

      <label className="form-field">
        <span>??</span>
        <textarea
          rows={2}
          placeholder="?????? 60 ?????"
          value={values.notes}
          onChange={(e) => handleChange('notes', e.target.value)}
        />
      </label>

      <button type="submit" className="primary-btn" disabled={!isValid}>
        ????
      </button>
    </form>
  );
}

export default JobForm;
