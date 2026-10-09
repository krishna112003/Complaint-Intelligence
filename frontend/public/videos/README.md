# Background Video Assets

Place your background video file here to enable an ambient video background on the frontend:

- **File Name**: `background.mp4` (or `background.webm`)
- **Recommended Settings**:
  - Resolution: 1080p (1920x1080) or 720p (1280x720)
  - Codec: H.264 (MP4) or VP9 (WebM)
  - Framerate: 24fps or 30fps
  - Bitrate: 1.5 - 3.5 Mbps (keeps loading instant and smooth)
  - Audio: None / Muted (browsers require background videos to be muted for autoplay)
  - Style: Dark abstract, flowing gradient mesh, subtle particles, or digital network loops (e.g. from Pexels, Pixabay, or Midjourney/Runway)

The `<BackgroundVideo />` component automatically detects `background.mp4`, loops it continuously using hardware acceleration, and applies a dark frosted vignette so all metrics, charts, and text remain 100% crisp and readable.
