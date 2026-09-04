import { useEffect, useRef, useState } from "react";
import UploadForm from "./components/UploadForm.jsx";
import StatusTimeline from "./components/StatusTimeline.jsx";
import VideoPreview from "./components/VideoPreview.jsx";
import { createJob, getJob } from "./api.js";

export default function App() {
  const [job, setJob] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState("");
  const pollRef = useRef(null);

  useEffect(() => {
    return () => clearInterval(pollRef.current);
  }, []);

  async function handleSubmit(payload) {
    setSubmitError("");
    setSubmitting(true);
    clearInterval(pollRef.current);
    try {
      const created = await createJob(payload);
      setJob(created);
      pollRef.current = setInterval(async () => {
        try {
          const updated = await getJob(created.job_id);
          setJob(updated);
          if (["completed", "failed"].includes(updated.status)) {
            clearInterval(pollRef.current);
          }
        } catch (err) {
          clearInterval(pollRef.current);
          setSubmitError(String(err.message || err));
        }
      }, 1500);
    } catch (err) {
      setSubmitError(String(err.message || err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="app">
      <div className="header">
        <h1>LightNoteAI</h1>
        <span className="badge">AI Video Object Editor</span>
      </div>
      <p className="subtitle">
        Upload a video, describe the edit in plain English, and watch an object get detected, tracked, and replaced.
      </p>

      <div className="layout">
        <UploadForm onSubmit={handleSubmit} submitting={submitting} />
        <div>
          {submitError && <div className="error-box" style={{ marginBottom: 16 }}>{submitError}</div>}
          <div style={{ display: "grid", gap: 20 }}>
            <StatusTimeline job={job} />
            <VideoPreview job={job} />
          </div>
        </div>
      </div>
    </div>
  );
}
