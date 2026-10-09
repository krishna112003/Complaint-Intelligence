"""Generate a high-aesthetic, seamless looping ambient background video for the frontend.
Outputs directly to frontend/public/videos/background.mp4.
"""

import math
from pathlib import Path
import numpy as np
import imageio

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "frontend" / "public" / "videos" / "background.mp4"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

WIDTH = 1280
HEIGHT = 720
FPS = 24
DURATION_SEC = 5.0
TOTAL_FRAMES = int(FPS * DURATION_SEC) # 120 frames

print(f"Rendering {TOTAL_FRAMES} frames ({WIDTH}x{HEIGHT} @ {FPS}fps) to {OUTPUT_PATH}...")

# Create node orbits for seamless loop
np.random.seed(42)
NUM_NODES = 24
nodes = []
for i in range(NUM_NODES):
    cx = 0.15 + 0.7 * np.random.rand()
    cy = 0.2 + 0.6 * np.random.rand()
    rx = 0.08 + 0.12 * np.random.rand()
    ry = 0.05 + 0.10 * np.random.rand()
    color_choice = np.random.choice([0, 1, 2]) # 0: Amber, 1: Cyan, 2: Violet
    nodes.append({
        "cx": cx * WIDTH,
        "cy": cy * HEIGHT,
        "rx": rx * WIDTH,
        "ry": ry * HEIGHT,
        "phase": np.random.rand() * 2 * math.pi,
        "color_type": color_choice,
        "radius": 4 + 4 * np.random.rand()
    })

writer = imageio.get_writer(
    str(OUTPUT_PATH),
    fps=FPS,
    codec='libx264',
    format='FFMPEG',
    ffmpeg_params=['-pix_fmt', 'yuv420p', '-crf', '22', '-preset', 'fast']
)

# Grid coordinates for vectorized wave rendering
X, Y = np.meshgrid(np.linspace(0, 1, WIDTH), np.linspace(0, 1, HEIGHT))

for frame_idx in range(TOTAL_FRAMES):
    tau = 2 * math.pi * (frame_idx / TOTAL_FRAMES)
    
    # 1. Base dark obsidian canvas
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
    frame[:, :, 0] = 9.0   # R
    frame[:, :, 1] = 10.0  # G
    frame[:, :, 2] = 16.0  # B
    
    # 2. Flowing harmonic wave ribbons (Amber & Cyan)
    # Wave 1: Warm Amber / Gold ribbon
    w1_y = 0.45 + 0.14 * np.sin(2 * math.pi * X * 1.5 + tau) + 0.06 * np.cos(4 * math.pi * X - tau)
    dist1 = np.abs(Y - w1_y)
    glow1 = np.exp(-dist1 * 18.0) * 0.75
    frame[:, :, 0] += glow1 * 245.0
    frame[:, :, 1] += glow1 * 158.0
    frame[:, :, 2] += glow1 * 11.0
    
    # Wave 2: Electric Cyan ribbon
    w2_y = 0.58 + 0.16 * np.sin(2 * math.pi * X * 2.0 - tau + 1.2) + 0.05 * np.cos(3 * math.pi * X + tau)
    dist2 = np.abs(Y - w2_y)
    glow2 = np.exp(-dist2 * 20.0) * 0.7
    frame[:, :, 0] += glow2 * 6.0
    frame[:, :, 1] += glow2 * 182.0
    frame[:, :, 2] += glow2 * 212.0
    
    # Wave 3: Deep Violet ribbon
    w3_y = 0.32 + 0.12 * np.cos(2 * math.pi * X * 1.8 + tau * 2)
    dist3 = np.abs(Y - w3_y)
    glow3 = np.exp(-dist3 * 22.0) * 0.55
    frame[:, :, 0] += glow3 * 139.0
    frame[:, :, 1] += glow3 * 92.0
    frame[:, :, 2] += glow3 * 246.0

    # 3. Ambient Drifting Nodes
    node_positions = []
    for node in nodes:
        nx = node["cx"] + node["rx"] * math.sin(tau + node["phase"])
        ny = node["cy"] + node["ry"] * math.cos(tau + node["phase"])
        node_positions.append((nx, ny, node["color_type"], node["radius"]))
        
        # Soft radial Gaussian stamp
        ix, iy = int(nx), int(ny)
        r_box = int(node["radius"] * 4)
        x0 = max(0, ix - r_box)
        x1 = min(WIDTH, ix + r_box)
        y0 = max(0, iy - r_box)
        y1 = min(HEIGHT, iy + r_box)
        
        if x1 > x0 and y1 > y0:
            gx, gy = np.meshgrid(np.arange(x0, x1) - nx, np.arange(y0, y1) - ny)
            d2 = gx * gx + gy * gy
            sigma2 = 2 * (node["radius"] ** 2)
            glow_stamp = np.exp(-d2 / sigma2)
            
            if node["color_type"] == 0:   # Amber
                frame[y0:y1, x0:x1, 0] += glow_stamp * 245.0 * 0.8
                frame[y0:y1, x0:x1, 1] += glow_stamp * 158.0 * 0.8
                frame[y0:y1, x0:x1, 2] += glow_stamp * 11.0 * 0.8
            elif node["color_type"] == 1: # Cyan
                frame[y0:y1, x0:x1, 0] += glow_stamp * 6.0 * 0.8
                frame[y0:y1, x0:x1, 1] += glow_stamp * 182.0 * 0.8
                frame[y0:y1, x0:x1, 2] += glow_stamp * 212.0 * 0.8
            else:                         # Violet
                frame[y0:y1, x0:x1, 0] += glow_stamp * 147.0 * 0.8
                frame[y0:y1, x0:x1, 1] += glow_stamp * 51.0 * 0.8
                frame[y0:y1, x0:x1, 2] += glow_stamp * 234.0 * 0.8

    # Clamp and cast to uint8
    final_frame = np.clip(frame, 0, 255).astype(np.uint8)
    writer.append_data(final_frame)
    
    if (frame_idx + 1) % 30 == 0 or frame_idx == TOTAL_FRAMES - 1:
        print(f"Rendered {frame_idx + 1}/{TOTAL_FRAMES} frames ({(frame_idx + 1)/TOTAL_FRAMES*100:.0f}%)")

writer.close()
print(f"Video successfully written to {OUTPUT_PATH} (size: {OUTPUT_PATH.stat().st_size / 1024 / 1024:.2f} MB)")
