import { useState } from "react";

const EXAMPLES = [
  "Replace the Coca-Cola bottle with Pepsi",
  "Remove the bottle from the table",
  "Replace Coca-Cola with Pepsi",
];

export default function UploadForm({ onSubmit, submitting }) {
  const [videoMode, setVideoMode] = useState("upload");
  const [videoFile, setVideoFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [referenceImage, setReferenceImage] = useState(null);
  const [prompt, setPrompt] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    onSubmit({
      videoFile: videoMode === "upload" ? videoFile : null,
      videoUrl: videoMode === "url" ? videoUrl : null,
      referenceImage,
      prompt,
    });
  }

  const canSubmit =
    prompt.trim().length > 0 &&
    ((videoMode === "upload" && videoFile) || (videoMode === "url" && videoUrl.trim())) &&
    !submitting;

  return (
    <form className="panel" onSubmit={handleSubmit}>
      <h2>1. Input</h2>

      <div className="field">
        <label>Video</label>
        <div className="tabs">
          <button type="button" className={videoMode === "upload" ? "active" : ""} onClick={() => setVideoMode("upload")}>
            Upload file
          </button>
          <button type="button" className={videoMode === "url" ? "active" : ""} onClick={() => setVideoMode("url")}>
            Video URL
          </button>
        </div>
        {videoMode === "upload" ? (
          <input type="file" accept="video/*" onChange={(e) => setVideoFile(e.target.files?.[0] || null)} />
        ) : (
          <input
            type="text"
            placeholder="https://example.com/clip.mp4"
            value={videoUrl}
            onChange={(e) => setVideoUrl(e.target.value)}
          />
        )}
        <div className="hint">Kept short for a CPU demo (auto-trimmed if longer than ~20s).</div>
      </div>

      <div className="field">
        <label>Reference image (optional)</label>
        <input type="file" accept="image/*" onChange={(e) => setReferenceImage(e.target.files?.[0] || null)} />
        <div className="hint">e.g. a photo of the Pepsi bottle to paste in. If omitted, we auto-generate one.</div>
      </div>

      <div className="field">
        <label>Editing prompt</label>
        <textarea
          placeholder='e.g. "Replace the Coca-Cola bottle with Pepsi"'
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
        />
        <div className="hint">
          Try:{" "}
          {EXAMPLES.map((ex, i) => (
            <span key={ex}>
              <button type="button" onClick={() => setPrompt(ex)} style={{ background: "none", border: "none", color: "#6d8bff", cursor: "pointer", padding: 0, fontSize: 11.5 }}>
                {ex}
              </button>
              {i < EXAMPLES.length - 1 ? " · " : ""}
            </span>
          ))}
        </div>
      </div>

      <button className="submit-btn" type="submit" disabled={!canSubmit}>
        {submitting ? "Starting…" : "Start Processing"}
      </button>
    </form>
  );
}
