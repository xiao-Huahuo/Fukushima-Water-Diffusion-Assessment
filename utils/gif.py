import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ==========================
# Config
# ==========================
FILE = "../outputs/mission1/H3/H3.nc"
OUTPUT = "../outputs/figures/gif/H3_surface_animation.gif"  # GIF 输出，ffmpeg 可选
DEPTH_LEVEL = 0
USE_LOG = True

# ==========================
# Load dataset
# ==========================
ds = xr.open_dataset(FILE)
C = ds["concentration"]  # (time, depth, lat, lon)
lat = ds["latitude"]
lon = ds["longitude"]

# ==========================
# Prepare 2D slices
# ==========================
C2D_all = C.isel(depth=DEPTH_LEVEL)  # (time, lat, lon)

if USE_LOG:
    C2D_all = np.log10(C2D_all + 1e-20)
    cmap = "plasma"
    cb_label = "log10(H3 concentration) [Bq/m^3]"
else:
    cmap = "viridis"
    cb_label = "H3 concentration [Bq/m^3]"

# ==========================
# Set up plot
# ==========================
fig, ax = plt.subplots(figsize=(11, 6))
pcm = ax.pcolormesh(lon, lat, C2D_all.isel(time=0), shading="auto", cmap=cmap)
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label(cb_label)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
title = ax.set_title("H3 Concentration (time=0)")

# ==========================
# Animation function
# ==========================
def update(frame):
    ax.clear()
    pcm = ax.pcolormesh(lon, lat, C2D_all.isel(time=frame), shading="auto", cmap=cmap)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"H3 Concentration (time step={frame})")
    return pcm,

# ==========================
# Create animation
# ==========================
ani = animation.FuncAnimation(
    fig, update, frames=C2D_all.shape[0], blit=False
)

# ==========================
# Save animation
# ==========================
ani.save(OUTPUT, writer="pillow", fps=10, dpi=150)
print("Animation saved to:", OUTPUT)
