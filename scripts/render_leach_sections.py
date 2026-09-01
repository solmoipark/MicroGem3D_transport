"""Cross-section renderings of the RT-W4.2b leaching run.

64^3 at 1 um, OPC, 168 h sealed then z-low face exposed to aerated water,
D0 = 2.0e-9 m2/s, dt = 0.4 s, 0.27 h of exposure (ckpt_007).
Dense arrays are (z, y, x); z = 0 is the exposed face (config.py:470).
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path(r"C:/Users/solmo/TINN_ reactive transport/runs"
           r"/leach_w4_opc64_dt0.4s_d0_2.0em09_u0.27h")
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
YSEC = 32

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
                     "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.linewidth": 0.6,
                     "savefig.dpi": 300})

MICRO_COLORS = np.array([[0.42, 0.31, 0.22], [0.85, 0.72, 0.52],
                         [0.72, 0.85, 0.95], [0.97, 0.97, 0.97]])
CLINKER = ("C3S", "C2S", "C3A", "C4AF", "inert", "gypsum",
           "hemihydrate", "anhydrite", "arcanite", "thenardite")


def read(ck, name):
    m = json.loads((ck / "arrays" / name / ".zarray").read_text())
    chunk = ck / "arrays" / name / ".".join(["0"] * len(m["shape"]))
    return np.frombuffer(chunk.read_bytes(),
                         dtype=np.dtype(m["dtype"])).reshape(m["shape"])


def hdr(ck):
    return json.loads((ck / "header.json").read_text())


def classes(ck, h):
    idx = [i for i, p in enumerate(h["solid_phase_ids"]) if p in CLINKER]
    return np.argmax(np.stack([read(ck, "anhydrous_fraction")[idx].sum(axis=0),
                               read(ck, "hydrate_fraction").sum(axis=0),
                               read(ck, "capillary_liquid"),
                               read(ck, "capillary_gas")]), axis=0)


def phase(ck, h, name):
    return read(ck, "hydrate_fraction")[h["hydrate_phase_ids"].index(name)]


ck0, ck1 = RUN / "ckpt_001", RUN / "ckpt_007"     # sealed 168.0 h, leached
h0, h1 = hdr(ck0), hdr(ck1)
N = h1["config"]["rve"]["grid_size"]
vox = h1["config"]["rve"]["voxel_size_um"]
tl = h1["time_h"] - h0["time_h"]
EXT = [0, N * vox, 0, N * vox]
TITLE = (f"OPC {N}$^3$ leaching, $D_0$ = 2.0" + r"$\times$" + "10$^{-9}$"
         f" m$^2$/s, {tl:.2f} h exposure")

ch0, ch1 = phase(ck0, h0, "Portlandite"), phase(ck1, h1, "Portlandite")
p0, p1 = ch0.mean(axis=(1, 2)), ch1.mean(axis=(1, 2))
front = 36.0 * vox   # front_vox recorded for t_leach = 0.27 h in
                     # runs/leach_w4_results_opc64_dt0.4s_d0_2.0em09_u0.27h.json
print(f"recorded CH front at z = {front:.0f} um; sealed mean CH "
      f"{p0.mean():.4f}, leached CH at z=0..10 {p1[:10].mean():.4f}")

# ------------------------------------------------ Fig A: microstructure
fig = plt.figure(figsize=(7.4, 3.6))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.28,
                      left=0.085, right=0.915, top=0.80, bottom=0.245)
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
cax = fig.add_subplot(gs[0, 3])
ax = axes[0]
ax.imshow(MICRO_COLORS[classes(ck1, h1)[:, YSEC, :]], origin="lower",
          interpolation="nearest", extent=EXT)
ax.set_title(f"Microstructure at {tl:.2f} h")
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c, ec="0.4", lw=0.4)
           for c in MICRO_COLORS],
          ["clinker", "hydrates", "capillary water", "gas"],
          loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2,
          frameon=False, handlelength=1.1)
vmax = float(max(ch0[:, YSEC, :].max(), ch1[:, YSEC, :].max()))
for ax, f, t in ((axes[1], ch0, "Portlandite, sealed (0 h)"),
                 (axes[2], ch1, f"Portlandite at {tl:.2f} h")):
    im = ax.imshow(f[:, YSEC, :], origin="lower", interpolation="nearest",
                   cmap="magma", vmin=0.0, vmax=vmax, extent=EXT)
    ax.set_title(t)
    ax.tick_params(labelleft=False)
axes[2].axhline(front, color="#4dd0e1", lw=1.2, ls="--")
axes[2].text(N * vox - 1.5, front + 1.5, f"front {front:.0f} µm",
             color="#4dd0e1", fontsize=7.5, ha="right", va="bottom")
cb = fig.colorbar(im, cax=cax)
cb.set_label("CH volume fraction (–)")
for ax in axes:
    ax.set_xlabel("x (µm)")
    ax.axhline(0.35, color="#1f77b4", lw=2.5, solid_capstyle="butt")
axes[0].set_ylabel("Distance from exposed face, z (µm)")
fig.suptitle(TITLE, fontsize=9.5, y=0.985)
fig.text(0.5, 0.895, "blue bar = exposed face; y = 32 µm section",
         ha="center", fontsize=7.5, color="0.35")
fig.savefig(OUT / "leach_microstructure_section.png")
fig.savefig(OUT / "leach_microstructure_section.pdf")
plt.close(fig)
print("wrote leach_microstructure_section")

# ------------------------------------------------------ Fig B: Ca/Si
el = h1["element_ids"]
ee = np.array(h1["endmember_elements"], float)
idx = [i for i, (p, _) in enumerate(h1["endmember_ids"]) if p == "CSHQ"]
em = read(ck1, "cluster_endmember_mol")
ca = em[:, idx] @ ee[idx][:, el.index("Ca")]
si = em[:, idx] @ ee[idx][:, el.index("Si")]
cid = read(ck1, "cluster_id")
zz = np.broadcast_to(np.arange(N)[:, None, None], cid.shape)
m = cid >= 0
nvox = np.bincount(cid[m].ravel(), minlength=em.shape[0])
zsum = np.bincount(cid[m].ravel(), weights=zz[m].ravel().astype(float),
                   minlength=em.shape[0])
zdom = np.where(nvox > 0, zsum / np.maximum(nvox, 1), np.nan)
ok = (si > 1e-15) & (nvox > 0)
Z, R, S = zdom[ok] * vox, (ca / si)[ok], si[ok]
si_tot = h1["hydrate_elements_ch"][h1["hydrate_phase_ids"].index("CSHQ")][
    el.index("Si")]
print(f"covered reactors {ok.sum()}/{em.shape[0]}; pooled Si = "
      f"{S.sum() / si_tot * 100:.1f} % of the C-S-H Si inventory")
print(f"Ca/Si: bulk {ca.sum() / si.sum():.4f}, per-domain "
      f"{R.min():.3f}..{R.max():.3f}")

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
ax = axes[0]
ax.scatter(R, Z, s=6 + 90 * S / S.max(), c="#2f6db3", alpha=0.75,
           edgecolors="none")
ax.axhline(front, color="#d62728", lw=1.2, ls="--")
ax.text(ax.get_xlim()[0], front + 1.2, "  CH front", color="#d62728",
        fontsize=7.5, va="bottom")
ax.set_xlabel("C-S-H Ca/Si (molar)")
ax.set_ylabel("Distance from exposed face, z (µm)")
ax.set_ylim(-1, N * vox)
ax.set_title("Per-domain Ca/Si vs depth\n(marker area $\propto$ pooled Si)")

ax = axes[1]
ax.hist(R, bins=np.linspace(0.85, 1.70, 35), weights=S, color="#2f6db3",
        edgecolor="white", linewidth=0.4)
ax.set_yscale("log")
ax.set_xlabel("C-S-H Ca/Si (molar)")
ax.set_ylabel("Pooled Si (mol, log scale)")
ax.set_title("Si-weighted distribution")
ax.axvline(ca.sum() / si.sum(), color="#d62728", lw=1.2)
ax.text(ca.sum() / si.sum() - 0.02, ax.get_ylim()[1] * 0.4,
        f"bulk {ca.sum() / si.sum():.3f}", color="#d62728", fontsize=7.5,
        ha="right", va="top", rotation=90)
fig.suptitle(TITLE, fontsize=9)
fig.text(0.5, 0.005,
         f"Domain-resolved endmember pools cover {ok.sum()} of {em.shape[0]} "
         f"equilibrium domains ({S.sum() / si_tot * 100:.0f} % of C-S-H Si); "
         "uncovered mass carries the global composition by construction.",
         ha="center", fontsize=7, color="0.35")
fig.tight_layout(rect=(0, 0.055, 1, 0.945))
fig.savefig(OUT / "leach_csh_casi.png")
fig.savefig(OUT / "leach_csh_casi.pdf")
plt.close(fig)
print("wrote leach_csh_casi")
