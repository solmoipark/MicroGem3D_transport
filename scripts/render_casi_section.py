"""C-S-H Ca/Si on the microstructure cross-section, RT-W4.2b leaching run.

Only voxels whose equilibrium domain carries its own endmember pool are
colour-mapped. C-S-H outside those domains takes the global composition by
construction (PRD v3.0/E2 covered-portion rule), so it is drawn flat grey
rather than coloured, which would imply a resolved measurement.
Dense arrays are (z, y, x); z = 0 is the exposed face.
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import ndimage

RUN = Path(r"C:/Users/solmo/TINN_ reactive transport/runs"
           r"/leach_w4_opc64_dt0.4s_d0_2.0em09_u0.27h")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
YSEC, FRONT = 32, 36.0
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "savefig.dpi": 300})
ck = RUN / "ckpt_007"
h = json.loads((ck / "header.json").read_text())


def read(name):
    m = json.loads((ck / "arrays" / name / ".zarray").read_text())
    chunk = ck / "arrays" / name / ".".join(["0"] * len(m["shape"]))
    return np.frombuffer(chunk.read_bytes(),
                         dtype=np.dtype(m["dtype"])).reshape(m["shape"])


el = h["element_ids"]
ee = np.array(h["endmember_elements"], float)
idx = [i for i, (p, _) in enumerate(h["endmember_ids"]) if p == "CSHQ"]
em = read("cluster_endmember_mol")
ca = em[:, idx] @ ee[idx][:, el.index("Ca")]
si = em[:, idx] @ ee[idx][:, el.index("Si")]
covered = si > 1e-15
ratio = np.where(covered, ca / np.where(si > 0, si, 1.0), np.nan)
bulk = ca.sum() / si.sum()

cid = read("cluster_id")
N = cid.shape[0]
vox = h["config"]["rve"]["voxel_size_um"]
TZ = h["config"]["transport"]["domains"]["tile_zyx"][0]
rid = np.full(cid.shape, -1, np.int64)
for z0 in range(0, N, TZ):            # domains are cluster ∩ z-band: stay in band
    sl = slice(z0, min(z0 + TZ, N))
    sub = cid[sl]
    if (sub >= 0).any():
        ind = ndimage.distance_transform_edt(sub < 0, return_distances=False,
                                             return_indices=True)
        rid[sl] = sub[tuple(ind)]

csh = read("hydrate_fraction")[h["hydrate_phase_ids"].index("CSHQ")]
has = csh > 0.02
safe = np.clip(rid, 0, None)
cov = (rid >= 0) & covered[safe] & has
field = np.where(cov, ratio[safe], np.nan)

lo, hi = 0.90, 1.65
cmap = plt.get_cmap("viridis")
norm = Normalize(lo, hi)
sec_f, sec_h, sec_c = field[:, YSEC, :], has[:, YSEC, :], cov[:, YSEC, :]
rgb = np.ones(sec_h.shape + (3,))                       # no C-S-H -> white
rgb[sec_h] = (0.80, 0.80, 0.80)                         # fallback -> grey
rgb[sec_c] = cmap(norm(sec_f[sec_c]))[:, :3]
print(f"section y={YSEC}: C-S-H {sec_h.sum()}, domain-resolved {sec_c.sum()} "
      f"({sec_c.sum() / sec_h.sum() * 100:.0f} %)")

fig = plt.figure(figsize=(7.5, 4.1))
TOP, BOT = 0.795, 0.255
gsL = fig.add_gridspec(1, 2, width_ratios=[1, 0.045], wspace=0.06,
                       left=0.085, right=0.475, top=TOP, bottom=BOT)
gsR = fig.add_gridspec(1, 1, left=0.655, right=0.965, top=TOP, bottom=BOT)

ax = fig.add_subplot(gsL[0, 0])
ax.imshow(rgb, origin="lower", interpolation="nearest",
          extent=[0, N * vox, 0, N * vox])
ax.axhline(FRONT, color="#d62728", lw=1.2, ls="--")
ax.text(N * vox - 1.5, FRONT + 1.2, "CH front 36 µm", color="#d62728",
        fontsize=7.5, ha="right", va="bottom")
ax.axhline(0.35, color="#1f77b4", lw=2.5, solid_capstyle="butt")
ax.set_xlabel("x (µm)")
ax.set_ylabel("Distance from exposed face, z (µm)")
ax.set_title(f"C-S-H Ca/Si, y = {YSEC} µm section", pad=6)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc="0.8", ec="0.45", lw=0.4),
           plt.Rectangle((0, 0), 1, 1, fc="w", ec="0.45", lw=0.4),
           plt.Line2D([], [], color="#1f77b4", lw=2.5)],
          [f"global {bulk:.3f} (not domain-resolved)", "no C-S-H",
           "exposed face"],
          loc="upper left", bbox_to_anchor=(-0.02, -0.20), ncol=1,
          frameon=False, handlelength=1.1, labelspacing=0.35)
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                  cax=fig.add_subplot(gsL[0, 1]))
cb.set_label("Ca/Si (molar), domain-resolved", fontsize=8)
cb.ax.tick_params(labelsize=7.5)

ax = fig.add_subplot(gsR[0, 0])
z = (np.arange(N) + 0.5) * vox
prof = np.array([np.nanmean(field[k]) if np.isfinite(field[k]).any() else np.nan
                 for k in range(N)])
ncov = cov.reshape(N, -1).sum(axis=1)
ax2 = ax.twiny()
ax2.fill_betweenx(z, 0, ncov, color="#c9a227", alpha=0.18, lw=0)
ax2.plot(ncov, z, "-", color="#c9a227", lw=0.9)
ax2.set_xlim(0, max(ncov.max(), 1) * 1.05)
ax2.set_xlabel("resolved voxels per layer", color="#a4801f", fontsize=7.5,
               labelpad=2)
ax2.tick_params(axis="x", labelsize=7, colors="#a4801f")
ax.plot(prof, z, "o-", ms=3, color="#2f6db3", lw=1.3, zorder=3)
ax.axvline(bulk, color="0.55", lw=1.0, ls=":")
ax.text(bulk - 0.02, 6, f"global {bulk:.3f}", color="0.45", fontsize=7.5,
        rotation=90, ha="right", va="bottom")
ax.axhline(FRONT, color="#d62728", lw=1.2, ls="--")
ax.set_xlabel("C-S-H Ca/Si (molar)")
ax.set_ylabel("Distance from exposed face, z (µm)")
ax.set_ylim(0, N * vox)
ax.set_xlim(lo, hi)
ax.set_title("Depth profile, resolved voxels only", pad=6)
ax.set_zorder(ax2.get_zorder() + 1)
ax.patch.set_visible(False)

fig.suptitle("OPC 64$^3$ leaching, $D_0$ = 2.0" + r"$\times$" + "10$^{-9}$"
             " m$^2$/s, 0.27 h exposure", fontsize=9.5, y=0.985)
fig.text(0.5, 0.915, f"{cov.sum()} of {has.sum()} C-S-H voxels carry a "
         f"domain-resolved composition ({cov.sum() / has.sum() * 100:.0f} %); "
         "the rest take the global value by construction",
         ha="center", fontsize=7.5, color="0.35")
fig.savefig(OUT / "leach_casi_section.png")
fig.savefig(OUT / "leach_casi_section.pdf")
print("wrote leach_casi_section")
print("resolved values:", np.unique(np.round(field[np.isfinite(field)], 4)))
for k in range(24, 42):
    if np.isfinite(prof[k]):
        print(f"  z={k:2d} um  Ca/Si {prof[k]:.3f}  ({ncov[k]} voxels)")
