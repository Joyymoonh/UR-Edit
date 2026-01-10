# UR-Edit 前端与接口指南

本说明位于 `frontend/` 目录，面向负责 UI/联调的同学，讲解 React 前端的安装方式以及如何对接 FastAPI 后端。

## 1. 技术栈与目录结构
- Vite + React + TypeScript（react-swc 模板）
- 样式文件：`src/style.css`
- 应用入口：`src/App.tsx`
- 默认 Mock API：`src/api/mockVideoEdit.ts`（联调时需替换为真实 API）

结构如下：
```
frontend/
  src/
    api/mockVideoEdit.ts
    components/JobCard.tsx
    components/JobForm.tsx
    App.tsx
    main.tsx
    style.css
    types.ts
  index.html
  package.json
  tsconfig.json
  README.md
```

## 2. 安装依赖
需要 Node.js 18 或以上。
```bash
cd frontend
npm install
```

## 3. 启动 / 构建
- 开发：
  ```bash
  npm run dev
  # 默认地址 http://localhost:5173
  ```
- 打包/预览：
  ```bash
  npm run build
  npm run preview
  ```

## 4. 切换到真实 API
1. 新建 `src/api/videoEdit.ts`：
   ```ts
   import type { VideoEditRequest } from '../types';

   const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://SERVER:8000';

   export async function runVideoEdit(payload: VideoEditRequest) {
     const res = await fetch(`${API_BASE}/jobs`, {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(payload),
     });
     if (!res.ok) throw new Error(await res.text());
     return res.json();
   }

   export async function getVideoEditStatus(jobId: string) {
     const res = await fetch(`${API_BASE}/jobs/${jobId}`);
     if (!res.ok) throw new Error(await res.text());
     return res.json();
   }
   ```
2. 根目录添加 `.env`：`VITE_API_BASE=http://server-ip:8000`。
3. 将 `src/App.tsx` 中的 `mockVideoEdit` 导入改为上述真实 API。

## 5. 与后端协作流程
1. **素材准备**：确保输入视频已经上传到服务器可访问路径（例如 `/data/uploads/foo.mp4`）。
2. **提交任务**：表单提交时构造 `VideoEditRequest` JSON，并将服务器路径填入 `input_video_path`。
3. **轮询状态**：沿用现有 `useEffect` 轮询逻辑，改为请求 `/jobs/{id}` 获取进度、日志和输出路径。
4. **结果展示**：当状态为 `finished` 时，使用返回的 `output_video_path` / `mask_preview_path` 给用户展示或提供下载。

> **接入真实接口前需与后端确认：**
> - FastAPI 服务地址（示例：`http://10.0.0.5:8000`）。
> - 上传接口（若已提供）：URL、请求方式、返回字段（如 `serverPath`）。
> - `POST /jobs` 的字段是否与 `VideoEditRequest` 完全一致，有无额外 token / 模型路径等要求。
> - `GET /jobs/{id}` 返回的 `output_video_path` / `mask_preview_path` 如何访问（绝对路径、下载接口或共享存储 URL）。
> 根据这些信息，将 `mockUpload` / `mockVideoEdit` 替换为真实 fetch 调用即可。

## 6. 额外提示
- 本地需要代理后端时，可在 `vite.config.ts` 配置 `server.proxy`。
- 若要支持文件上传，可在后端新增 `/upload` 接口或接入对象存储，前端上传成功后再调用 `/jobs`。
- 构建产物位于 `frontend/dist`，可部署到 Nginx、Vercel 等任意静态托管平台。

后端/FastAPI 的部署细节请参考仓库根目录 README。
