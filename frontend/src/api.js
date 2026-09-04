// In dev, Vite proxies /api and /media to the backend (see vite.config.js), so
// BASE stays empty. For a production build served separately from the API,
// set VITE_API_BASE_URL (e.g. https://api.yourdomain.com) at build time.
const BASE = import.meta.env.VITE_API_BASE_URL || "";

export async function createJob({ videoFile, videoUrl, referenceImage, prompt }) {
  const form = new FormData();
  form.append("prompt", prompt);
  if (videoFile) form.append("video", videoFile);
  if (videoUrl) form.append("video_url", videoUrl);
  if (referenceImage) form.append("reference_image", referenceImage);

  const res = await fetch(`${BASE}/api/jobs`, { method: "POST", body: form });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

export function mediaUrl(path) {
  if (!path) return null;
  return `${BASE}${path}`;
}

export async function getJob(jobId) {
  const res = await fetch(`${BASE}/api/jobs/${jobId}`);
  if (!res.ok) throw new Error(`Failed to fetch job ${jobId}`);
  return res.json();
}
