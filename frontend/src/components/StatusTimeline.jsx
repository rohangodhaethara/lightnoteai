const STEPS = [
  { key: "queued", label: "Queued" },
  { key: "parsing_instruction", label: "Parsing instruction (LLM)" },
  { key: "detecting_object", label: "Extracting frames" },
  { key: "tracking_object", label: "Locating & tracking object" },
  { key: "editing_frames", label: "Editing frames" },
  { key: "rendering_video", label: "Rendering video" },
  { key: "completed", label: "Completed" },
];

export default function StatusTimeline({ job }) {
  if (!job) return null;
  const currentIndex = STEPS.findIndex((s) => s.key === job.status);
  const failed = job.status === "failed";

  return (
    <div className="panel">
      <h2>2. Processing Status</h2>

      <div className="progress-label">{job.progress}%</div>
      <div className="progress-bar">
        <div className="progress-bar-fill" style={{ width: `${job.progress}%` }} />
      </div>

      <div className="status-steps" style={{ marginTop: 16 }}>
        {STEPS.map((s, i) => {
          let cls = "step";
          if (failed && i === currentIndex) cls += " failed";
          else if (i < currentIndex || (i === currentIndex && job.status === "completed")) cls += " done";
          else if (i === currentIndex) cls += " active";
          return (
            <div key={s.key} className={cls}>
              <span className="dot" />
              <span>{s.label}</span>
            </div>
          );
        })}
      </div>

      {job.message && <div className="message-line">{job.message}</div>}
      {job.error && <div className="error-box">Error: {job.error}</div>}

      {job.parsed_instruction && (
        <div className="parsed-box">
          <div className="row">
            <span className="k">Operation</span>
            <span className="v">{job.parsed_instruction.operation}</span>
          </div>
          <div className="row">
            <span className="k">Target</span>
            <span className="v">{job.parsed_instruction.target_object || "—"}</span>
          </div>
          {job.parsed_instruction.replacement_object && (
            <div className="row">
              <span className="k">Replacement</span>
              <span className="v">{job.parsed_instruction.replacement_object}</span>
            </div>
          )}
          {job.parsed_instruction.target_text && (
            <div className="row">
              <span className="k">Text: from → to</span>
              <span className="v">
                {job.parsed_instruction.target_text} → {job.parsed_instruction.replacement_text}
              </span>
            </div>
          )}
          <div className="row">
            <span className="k">Mapped vision class</span>
            <span className="v">{job.parsed_instruction.coco_class || "n/a (heuristic mode)"}</span>
          </div>
          <div className="row">
            <span className="k">Parsed by</span>
            <span className="v">{job.parsed_instruction.parsed_by}</span>
          </div>
        </div>
      )}
    </div>
  );
}
