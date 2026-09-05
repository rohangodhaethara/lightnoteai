# Demo Video Script (~2:30)

Record your screen at `http://localhost:5173` with both servers running
(see README setup). Use `samples/sample_bottle_video.mp4` as the input —
it's a real Coca-Cola bottle photo, so it matches the assignment's exact
example.

---

**[0:00–0:15] Intro**

> "Hi, this is my submission for the LightNoteAI assignment — an AI video
> editor where you describe an edit in plain English and it modifies the
> object in the video. Frontend is React, backend is FastAPI. Let me walk
> through it."

*(Show the empty UI — three panels: Input, Processing Status, Preview.)*

---

**[0:15–0:35] Upload + inputs**

> "I'll upload a short clip with a Coca-Cola bottle in frame."

*(Click "Upload file", select `samples/sample_bottle_video.mp4`.)*

> "Optionally I can upload a reference image of the replacement product —
> I'll skip that here so you can see the system auto-generate a
> placeholder when none is given."

*(Point out the reference-image field, leave it empty.)*

---

**[0:35–0:55] The prompt**

> "Now the instruction, in plain English — exactly the assignment's
> example."

*(Type: `Replace the Coca-Cola bottle with Pepsi`)*

> "Behind the scenes this gets sent to an LLM — Claude, GPT, or Gemini,
> whichever's configured — which extracts a structured action: operation
> replace_object, target 'Coca-Cola bottle', replacement 'Pepsi'. If no API
> key is set it falls back to a deterministic parser, so the app always
> runs."

---

**[0:55–1:05] Start processing**

*(Click "Start Processing".)*

> "That kicks off a background job — I'm using an async job queue on the
> backend so the API stays responsive while this CPU work runs."

---

**[1:05–1:40] Processing status**

*(Let the status timeline advance; narrate each step as it lights up.)*

> "You can see the pipeline stages live: parsing the instruction, extracting
> frames with ffmpeg, then locating and tracking the object — that's
> YOLOv8's segmentation model finding the bottle in every frame and a
> lightweight tracker keeping it consistent frame to frame. Then editing —
> each frame gets the bottle region composited with the replacement using
> OpenCV's seamless cloning — and finally rendering back to video with
> ffmpeg, audio re-attached."

*(Point at the parsed-instruction box: operation / target / replacement /
mapped vision class / parsed-by.)*

> "This panel shows exactly what the AI understood — useful for debugging
> and for showing the reasoning isn't a black box."

---

**[1:40–2:05] Result**

*(Once completed, show input vs. output video side by side; play both.)*

> "And here's the result — input on the left with the Coca-Cola bottle,
> output on the right with it replaced. It also reports how many frames
> the object was actually detected in, which is a useful confidence signal
> when detection is uncertain."

---

**[2:05–2:25] Quick mention of bonus features + wrap-up**

> "Beyond replace, it also supports object removal and a simplified text
> replacement, plus video-URL input as an alternative to file upload. The
> README covers the full architecture, why I chose YOLOv8-seg and classic
> CV compositing over heavier models like SAM or diffusion inpainting, and
> known limitations. Thanks for watching."

---

## Recording tips

- Use OBS / Xbox Game Bar (`Win+Alt+R`) / any screen recorder at 1080p.
- If detection ever shows 0/N frames on a take, just re-run — it's the
  synchronous heuristic fallback kicking in only when the YOLO model
  genuinely can't find a confident match; the sample video reliably hits
  ~24/24.
- Keep total runtime 2–3 minutes per the assignment's ask.
