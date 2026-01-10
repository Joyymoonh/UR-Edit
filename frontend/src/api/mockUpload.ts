const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export interface MockUploadResponse {
  serverPath: string;
}

export async function mockUpload(file: File, jobId: string): Promise<MockUploadResponse> {
  await delay(700 + Math.random() * 500);
  const safeName = file.name.replace(/\s+/g, '_');
  return {
    serverPath: `/mock_uploads/${jobId}_${safeName}`,
  };
}
