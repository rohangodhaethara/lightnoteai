import { mediaUrl } from "../api.js";

export default function VideoPreview({ job }) {
  if (!job) {
    return (
      <div className="panel">
        <h2>3. Preview</h2>
        <div className="empty-state">Upload a video and start processing to see input/output previews here.</div>
      </div>
    );
  }

  return (
    <div className="panel">
      <h2>3. Preview</h2>
      <div className="videos">
        <div className="video-card">
          <h3>Input video</h3>
          {job.input_video_url ? <video src={mediaUrl(job.input_video_url)} controls /> : <div className="empty-state">—</div>}
          {job.reference_image_url && (
            <div style={{ marginTop: 10 }}>
              <h3>Reference image</h3>
              <img src={mediaUrl(job.reference_image_url)} alt="reference" />
            </div>
          )}
        </div>
        <div className="video-card">
          <h3>Output video</h3>
          {job.output_video_url ? (
            <video src={mediaUrl(job.output_video_url)} controls />
          ) : (
            <div className="empty-state">{job.status === "failed" ? "Processing failed." : "Not ready yet…"}</div>
          )}
        </div>
      </div>
    </div>
  );
}
