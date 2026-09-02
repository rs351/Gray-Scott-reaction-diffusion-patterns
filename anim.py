#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

v = np.loadtxt(sys.argv[1])
y_s, x_s, t_end, t_int, n_cs = (int(v[i]) for i in range(5))

nc = y_s * x_s
per_frame = (3 + n_cs) * nc                 # eps, u_x, u_y, then n_cs species
n_frames = (len(v) - 5) // per_frame
body = v[5:5 + n_frames * per_frame].reshape(n_frames, 3 + n_cs, nc)

# field f, frame n -> (y_s, x_s); stored as j*y_s+i with j the x index
def fld(n, f):
    return body[n, f].reshape(x_s, y_s).T

titles = ["temperature", "horizontal velocity", "vertical velocity", "species A"]
idx = [0, 1, 2, 3]                          # species A is the first psi block

# fixed colour scales from the whole run so frames are comparable
lims = []
for f in idx:
    a = body[:, f, :]
    if f in (1, 2):
        m = np.abs(a).max()
        lims.append((-m, m, "RdBu_r"))
    elif f == 0:
        lims.append((a.min(), a.max(), "inferno"))
    else:
        lims.append((a.min(), a.max(), "viridis"))

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
ims = []
for ax, f, ttl, (lo, hi, cm) in zip(axes.flat, idx, titles, lims):
    im = ax.imshow(fld(0, f), origin="upper", aspect="equal", cmap=cm,
                   vmin=lo, vmax=hi, extent=[0, x_s, y_s, 0], interpolation="nearest")
    ax.set_title(ttl)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ims.append(im)

sup = fig.suptitle("t = 0")

def update(n):
    for im, f in zip(ims, idx):
        im.set_data(fld(n, f))
    sup.set_text(f"t = {n * t_int}")
    return ims + [sup]

anim = FuncAnimation(fig, update, frames=n_frames, interval=60, blit=False)
fig.tight_layout()

if len(sys.argv) > 2:
    anim.save(sys.argv[2], fps=15, dpi=110)
    print(f"wrote {sys.argv[2]} ({n_frames} frames)")
else:
    plt.show()