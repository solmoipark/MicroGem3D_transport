"""Read-only S4 analysis of the returned S3 production checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402


FOCUS_PHASES = (
    "CSHQ",
    "Portlandite",
    "ettringite",
    "Gypsum",
    "SO4_OH_AFm",
    "OH_SO4_AFm",
    "C3(AF)S0.84H",
)
SULFATE_CARRIERS = ("hemihydrate", "anhydrite")
CSHQ_HIST_EDGES = np.linspace(0.8, 2.2, 29)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def mmol_100g(mol: float, binder_mass_g: float) -> float:
    return float(mol) * 100_000.0 / binder_mass_g


def binder_mass(state, registry, kinetic_ids) -> float:
    reactive_mass = sum(
        float(state.initial_phase_mol[i]) * registry.get(pid).molar_mass_g_mol
        for i, pid in enumerate(kinetic_ids)
    )
    assigned = sum(float(v) for v in state.config.binder.mass_fractions.values())
    if assigned <= 0.0:
        raise ValueError("binder mass fractions have no assigned mass")
    return reactive_mass / assigned


def dense_morphology(state, phase_volume_fractions) -> dict:
    a = state.anhydrous_fraction.sum(axis=0)
    h = state.hydrate_fraction.sum(axis=0)
    p = state.capillary_liquid + state.capillary_gas
    am, hm, pm = a > 1e-12, h > 1e-12, p > 1e-12
    voxel = float(state.config.rve.voxel_size_um)
    volume = (state.grid_size * voxel) ** 3

    def interface_density(x, y):
        faces = 0
        for axis in range(3):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(0, -1)
            hi[axis] = slice(1, None)
            faces += int(
                np.count_nonzero(
                    (x[tuple(lo)] & y[tuple(hi)])
                    | (y[tuple(lo)] & x[tuple(hi)])
                )
            )
        return faces * voxel * voxel / volume

    return {
        "anhydrous_volume_fraction": float(a.mean()),
        "hydrate_envelope_volume_fraction": float(h.mean()),
        "capillary_volume_fraction_dense": float(p.mean()),
        "anhydrous_presence_fraction": float(am.mean()),
        "hydrate_presence_fraction": float(hm.mean()),
        "capillary_presence_fraction": float(pm.mean()),
        "anhydrous_hydrate_interface_area_per_um3": interface_density(am, hm),
        "hydrate_capillary_interface_area_per_um3": interface_density(hm, pm),
        "phase_volume_fractions": phase_volume_fractions(state),
    }


def cluster_statistics(state, cluster_ca_si, casi_channel) -> tuple[dict, dict]:
    ratios = cluster_ca_si(state)
    channel = casi_channel(state)
    if not ratios or channel is None:
        return {
            "authority_channel": channel,
            "cluster_count": 0,
            "histogram_edges_ca_si": CSHQ_HIST_EDGES.tolist(),
            "histogram_cluster_count": [0] * (len(CSHQ_HIST_EDGES) - 1),
        }, {}

    idxs = [
        j for j, (hydrate, _dc) in enumerate(state.endmember_ids)
        if hydrate == channel
    ]
    pools = np.clip(state.cluster_endmember_mol[:, idxs], 0.0, None)
    weights = pools.sum(axis=1)
    ids = np.array(sorted(ratios), dtype=np.int64)
    vals = np.array([ratios[int(i)] for i in ids], dtype=np.float64)
    w = weights[ids]
    hist_count, _ = np.histogram(vals, bins=CSHQ_HIST_EDGES)
    hist_weight, _ = np.histogram(vals, bins=CSHQ_HIST_EDGES, weights=w)
    wsum = float(w.sum())
    weighted_mean = float(np.average(vals, weights=w)) if wsum > 0 else None

    labels = state.cluster_id.ravel()
    valid = labels >= 0
    labels_valid = labels[valid].astype(np.int64)
    coords = np.indices(state.cluster_id.shape, dtype=np.float64).reshape(3, -1)[:, valid]
    size = max(int(labels_valid.max()) + 1 if labels_valid.size else 0, len(weights))
    counts = np.bincount(labels_valid, minlength=size)
    sums = [np.bincount(labels_valid, weights=coords[i], minlength=size) for i in range(3)]
    centroids = {}
    for cid in ids:
        if cid < len(counts) and counts[cid] > 0:
            centroids[int(cid)] = {
                "z": float(sums[0][cid] / counts[cid]),
                "y": float(sums[1][cid] / counts[cid]),
                "x": float(sums[2][cid] / counts[cid]),
                "voxel_count": int(counts[cid]),
                "ca_si": float(ratios[int(cid)]),
                "channel_pool_mol": float(weights[cid]),
            }

    stats = {
        "authority_channel": channel,
        "cluster_count": int(len(vals)),
        "min": float(vals.min()),
        "q05": float(np.quantile(vals, 0.05)),
        "median": float(np.median(vals)),
        "q95": float(np.quantile(vals, 0.95)),
        "max": float(vals.max()),
        "mean_unweighted": float(vals.mean()),
        "std_unweighted": float(vals.std()),
        "mean_pool_weighted": weighted_mean,
        "histogram_edges_ca_si": CSHQ_HIST_EDGES.tolist(),
        "histogram_cluster_count": hist_count.tolist(),
        "histogram_channel_pool_fraction": (
            (hist_weight / wsum).tolist() if wsum > 0 else hist_weight.tolist()
        ),
        "below_histogram": int(np.count_nonzero(vals < CSHQ_HIST_EDGES[0])),
        "above_histogram": int(np.count_nonzero(vals > CSHQ_HIST_EDGES[-1])),
    }
    return stats, centroids


def mip_proxy(psd: dict) -> list[dict]:
    """MIP-convention conversion of EDT diameters, not an intrusion simulation."""
    gamma = 0.485  # N/m, mercury
    theta = math.radians(140.0)
    edges = np.asarray(psd["edges_um"], dtype=float)
    frac = np.asarray(psd["volume_fraction_per_bin"], dtype=float)
    records = []
    for edge in reversed(edges):
        pressure_mpa = 4.0 * gamma * abs(math.cos(theta)) / (edge * 1e-6) / 1e6
        # At this pressure, bins whose characteristic upper diameter is at
        # least the threshold are counted as conventionally intruded.
        upper = edges
        cumulative = float(frac[upper >= edge].sum())
        records.append(
            {
                "pressure_MPa": pressure_mpa,
                "diameter_threshold_um": float(edge),
                "cumulative_capillary_volume_fraction": cumulative,
            }
        )
    return records


def plot_outputs(records: list[dict], outdir: Path, map_images: dict, centroids: dict) -> None:
    ages = np.array([r["time_h"] for r in records])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for phase in ("CSHQ", "Portlandite", "ettringite", "Gypsum"):
        axes[0, 0].plot(ages, [r["phase_mmol_per_100g"].get(phase, 0.0) for r in records], marker="o", label=phase)
    axes[0, 0].set(xlabel="Age (h)", ylabel="mmol / 100 g binder", xscale="log", title="Hydrate inventory")
    axes[0, 0].legend(fontsize=8)
    for key in ("free", "gel", "bound"):
        axes[0, 1].plot(ages, [r["water_g_per_100g"][key] for r in records], marker="o", label=key)
    axes[0, 1].set(xlabel="Age (h)", ylabel="g H2O / 100 g binder", xscale="log", title="Water compartments")
    axes[0, 1].legend(fontsize=8)
    axes[1, 0].plot(ages, [r["porosity_capillary"] for r in records], marker="o", label="capillary")
    axes[1, 0].plot(ages, [r["porosity_total"] for r in records], marker="o", label="total")
    axes[1, 0].plot(ages, [r["porosity_split"]["connected"] for r in records], marker="o", label="connected capillary")
    axes[1, 0].set(xlabel="Age (h)", ylabel="RVE fraction", xscale="log", title="Porosity")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(ages, [r["CSHQ_bulk_ca_si"] for r in records], marker="o", label="bulk")
    axes[1, 1].plot(ages, [r["CSHQ_cluster_ca_si"].get("mean_pool_weighted") for r in records], marker="o", label="cluster-pool weighted")
    axes[1, 1].set(xlabel="Age (h)", ylabel="Ca/Si", xscale="log", title="Authoritative CSHQ Ca/Si")
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "s3_production_timeseries.png", dpi=180)
    plt.close(fig)

    selected = [r for r in records if r["time_h"] in (24.0, 168.0, 672.0)]
    selected_values = [
        point["ca_si"]
        for rec in selected
        for point in centroids.get(str(int(rec["time_h"])), {}).values()
    ]
    if selected_values:
        zoom_lo, zoom_hi = np.quantile(selected_values, [0.005, 0.995])
        pad = max((zoom_hi - zoom_lo) * 0.08, 2e-5)
        zoom_lo, zoom_hi = float(zoom_lo - pad), float(zoom_hi + pad)
    else:
        zoom_lo, zoom_hi = 0.8, 2.2
    zoom_edges = np.linspace(zoom_lo, zoom_hi, 31)
    fig, axes = plt.subplots(1, len(selected), figsize=(13, 3.8), sharex=True, sharey=True)
    for ax, rec in zip(axes, selected):
        points = list(centroids.get(str(int(rec["time_h"])), {}).values())
        vals = np.asarray([p["ca_si"] for p in points], dtype=float)
        weights = np.asarray([p["channel_pool_mol"] for p in points], dtype=float)
        hist, _ = np.histogram(vals, bins=zoom_edges, weights=weights)
        frac = hist / hist.sum() if hist.sum() > 0 else hist
        ax.bar(zoom_edges[:-1], frac, width=np.diff(zoom_edges), align="edge")
        ax.set_title(f"{rec['time_h']:.0f} h")
        ax.set_xlabel("CSHQ Ca/Si")
    axes[0].set_ylabel("CSHQ pool fraction")
    fig.tight_layout()
    fig.suptitle(f"Pool-weighted cluster distribution (zoom: {zoom_lo:.5f}–{zoom_hi:.5f})")
    fig.savefig(outdir / "cshq_cluster_histograms_zoomed.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for rec in selected:
        psd = rec["pore_size_distribution"]
        edges = np.asarray([0.0] + psd["edges_um"])
        mids = np.where(edges[:-1] == 0, edges[1:] / 2, np.sqrt(edges[:-1] * edges[1:]))
        ax.step(mids, psd["volume_fraction_per_bin"], where="mid", label=f"{rec['time_h']:.0f} h")
    ax.set(xlabel="Local pore diameter (µm)", ylabel="Capillary-volume fraction per bin", xscale="log", title="Periodic EDT pore-size distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "pore_size_distributions.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(14, 4.5))
    for i, rec in enumerate(selected, start=1):
        age = str(int(rec["time_h"]))
        ax = fig.add_subplot(1, 3, i, projection="3d")
        points = centroids.get(age, {})
        if points:
            vals = list(points.values())
            sc = ax.scatter(
                [v["x"] for v in vals], [v["y"] for v in vals], [v["z"] for v in vals],
                c=[v["ca_si"] for v in vals], s=np.maximum(5, np.sqrt([v["voxel_count"] for v in vals])),
                vmin=zoom_lo, vmax=zoom_hi, cmap="coolwarm", alpha=0.8,
            )
            fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.08)
        ax.set_title(f"{age} h")
        ax.set_xlabel("x voxel")
        ax.set_ylabel("y voxel")
        ax.set_zlabel("z voxel")
    fig.suptitle(f"CSHQ cluster-centroid Ca/Si map (zoom: {zoom_lo:.5f}–{zoom_hi:.5f})")
    fig.tight_layout()
    fig.savefig(outdir / "cshq_cluster_centroids_3d_zoomed.png", dpi=180)
    plt.close(fig)

    for age, img in map_images.items():
        Image.fromarray(img).save(outdir / f"cshq_ca_si_central_slice_{age}h.png")


def report(payload: dict) -> str:
    lines = [
        "# S3 production run 분석 팩 (S4, checkpoint 기반)",
        "",
        "S3의 10개 checkpoint를 읽기 전용으로 분석했다. 신규 chemistry 계산은 없다.",
        "",
        "## 무결성 및 범위",
        "",
        f"- 최종 재령: {payload['qualification']['final_time_h']:.0f} h",
        f"- accepted/rejected: {payload['qualification']['accepted_steps']} / {payload['qualification']['rejected_trials']}",
        f"- checkpoint readback: {payload['qualification']['all_checkpoint_readbacks_match']}",
        f"- PC bundle 불변: {payload['qualification']['bundle_sha256_unchanged']}",
        "- CSHQ Ca/Si authority: cluster endmember pool의 CSHQ channel만 사용",
        "",
        "## 시계열 요약",
        "",
        "| Age (h) | CSHQ | Portlandite | Ettringite | Gypsum | Capillary porosity | Connected | CSHQ bulk Ca/Si | Cluster q05–q95 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["records"]:
        cs = row["CSHQ_cluster_ca_si"]
        qrange = f"{cs.get('q05', float('nan')):.4f}–{cs.get('q95', float('nan')):.4f}" if cs.get("q05") is not None else "n/a"
        lines.append(
            f"| {row['time_h']:.0f} | {row['phase_mmol_per_100g'].get('CSHQ', 0):.3f} | "
            f"{row['phase_mmol_per_100g'].get('Portlandite', 0):.3f} | "
            f"{row['phase_mmol_per_100g'].get('ettringite', 0):.3f} | "
            f"{row['phase_mmol_per_100g'].get('Gypsum', 0):.3f} | "
            f"{row['porosity_capillary']:.5f} | {row['porosity_split']['connected']:.5f} | "
            f"{row['CSHQ_bulk_ca_si']:.5f} | {qrange} |"
        )
    dep = payload["sulfate_depletion"]
    lines += [
        "",
        "## Sulfate 관찰",
        "",
        f"- 담체(hemihydrate+anhydrite) 1% 이하가 처음 관찰된 출력 재령: {dep['first_observed_carrier_below_1pct_h']} h",
        f"- 용존 S가 0.01 mmol/100 g 이하가 처음 관찰된 출력 재령: {dep['first_observed_aqueous_s_below_0p01_mmol_100g_h']} h",
        "- 이는 출력 checkpoint 사이의 관찰 구간이며 정확한 고갈 순간을 의미하지 않는다.",
        "",
        "## 제한사항과 재계산 판단",
        "",
        "현재 결과만으로 상·물 인벤토리, CSHQ Ca/Si 분포, 공극 연결성/PSD,",
        "MIP 관례 proxy와 sulfate 출력시점 변화는 분석 가능하다. 반면 S3 작업지시서가",
        "요구한 step별 unmet-release 증분, 동결 클러스터 수, secondary gypsum net flux는",
        "기록되지 않았다. 이 세 항목의 정확한 step 귀속이 본문 필수이면 로거 보강 후",
        "S3 재계산이 필요하고, checkpoint 기반 결과만 필요하면 재계산할 필요가 없다.",
        "",
        "`mip_convention_proxy`는 EDT pore-body 직경의 Washburn 환산이며 실제 mercury",
        "침입 또는 throat-network 시뮬레이션이 아니다.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    run = Path(args.run).resolve()
    outdir = Path(args.outdir).resolve()
    if outdir.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {outdir}")
    sys.path.insert(0, str(Path(args.source_root).resolve()))
    from tinn.analysis import (  # noqa: E402
        casi_channel,
        casi_map_rgb,
        cluster_ca_si,
        liquid_percolation,
        phase_volume_fractions,
        pore_size_distribution,
        porosity_split,
    )
    from tinn.registry import ELEMENT_IDS, KINETIC_PHASE_IDS, default_registry  # noqa: E402
    from tinn.storage import load_checkpoint  # noqa: E402

    qualification = read_json(run / "qualification.json")
    gates = {
        "final_time_672h": float(qualification["final_time_h"]) == 672.0,
        "accepted_steps_57": int(qualification["accepted_steps"]) == 57,
        "zero_rejected_trials": int(qualification["rejected_trials"]) == 0,
        "checkpoint_readback": bool(qualification["all_checkpoint_readbacks_match"]),
        "bundle_immutable": bool(qualification["bundle_sha256_unchanged"]),
        "rollback_preserved": bool(qualification["rollback_dense_state_preserved"]),
        "zero_boundary_exchange": all(float(v) == 0.0 for v in qualification["boundary_exchange_mol_by_element"].values()),
    }
    if not all(gates.values()):
        raise RuntimeError(f"S3 qualification gate failed: {gates}")

    summary_rows = {float(r["time_h"]): r for r in read_json(run / "summary.json")["outputs"]}
    checkpoints = []
    for ckpt in run.glob("ckpt_*"):
        header = read_json(ckpt / "header.json")
        checkpoints.append((float(header["time_h"]), ckpt))
    checkpoints.sort()
    if [age for age, _ in checkpoints] != sorted(summary_rows):
        raise RuntimeError("checkpoint and summary output ages do not match")

    outdir.mkdir(parents=True)
    registry = default_registry()
    records = []
    map_images = {}
    all_centroids = {}
    initial_carrier = None
    binder_g = None
    for age, checkpoint in checkpoints:
        state = load_checkpoint(str(checkpoint), registry)
        if not math.isclose(float(state.time_h), age, abs_tol=1e-12):
            raise RuntimeError(f"checkpoint header/state time mismatch at {checkpoint}")
        if binder_g is None:
            binder_g = binder_mass(state, registry, KINETIC_PHASE_IDS)
        summary = summary_rows[age]
        phase = {
            key: mmol_100g(value, binder_g)
            for key, value in summary["hydrate_mol"].items()
        }
        remaining = {
            key: mmol_100g(value, binder_g)
            for key, value in summary["phase_mol"].items()
        }
        water = {
            "free": float(state.water_free_mol) * registry.get("H2O").molar_mass_g_mol * 100.0 / binder_g,
            "gel": float(state.water_gel_mol) * registry.get("H2O").molar_mass_g_mol * 100.0 / binder_g,
            "bound": float(state.water_bound_mol) * registry.get("H2O").molar_mass_g_mol * 100.0 / binder_g,
        }
        carrier = sum(remaining.get(key, 0.0) for key in SULFATE_CARRIERS)
        if initial_carrier is None:
            initial_carrier = sum(
                mmol_100g(state.initial_phase_mol[i], binder_g)
                for i, pid in enumerate(KINETIC_PHASE_IDS)
                if pid in SULFATE_CARRIERS
            )
        s_index = ELEMENT_IDS.index("S")
        aqueous_s = mmol_100g(float(state.cluster_inventory[:, s_index].sum()), binder_g)
        cs_stats, centroids = cluster_statistics(state, cluster_ca_si, casi_channel)
        psd = pore_size_distribution(state)
        split = porosity_split(state)
        cshq_bulk = summary.get("solid_solution_composition", {}).get("CSHQ", {}).get("ca_si")
        rec = {
            "time_h": age,
            "accept_count": int(state.accept_count),
            "phase_mmol_per_100g": phase,
            "remaining_anhydrous_mmol_per_100g": remaining,
            "reaction_degree": {k: float(v) for k, v in summary["alpha"].items()},
            "water_g_per_100g": water,
            "porosity_capillary": float(summary["porosity_capillary"]),
            "porosity_total": float(summary["porosity_total"]),
            "porosity_split": split,
            "liquid_percolation": liquid_percolation(state.capillary_liquid),
            "pore_size_distribution": psd,
            "mip_convention_proxy": mip_proxy(psd),
            "CSHQ_bulk_ca_si": safe(cshq_bulk),
            "CSHQ_cluster_ca_si": cs_stats,
            "sulfate": {
                "remaining_primary_carriers_mmol_per_100g": carrier,
                "aqueous_s_mmol_per_100g": aqueous_s,
                "gypsum_mmol_per_100g": phase.get("Gypsum", 0.0),
                "ettringite_mmol_per_100g": phase.get("ettringite", 0.0),
                "sulfate_afm_mmol_per_100g": phase.get("SO4_OH_AFm", 0.0) + phase.get("OH_SO4_AFm", 0.0),
            },
            "morphology": dense_morphology(state, phase_volume_fractions),
        }
        records.append(rec)
        all_centroids[str(int(age))] = centroids
        img = casi_map_rgb(state, lo=0.8, hi=2.2)
        if img is not None:
            map_images[str(int(age))] = img
        del state
        gc.collect()

    carrier_depletion = next(
        (r["time_h"] for r in records if r["sulfate"]["remaining_primary_carriers_mmol_per_100g"] <= 0.01 * initial_carrier),
        None,
    )
    aqueous_depletion = next(
        (r["time_h"] for r in records if r["sulfate"]["aqueous_s_mmol_per_100g"] <= 0.01),
        None,
    )
    payload = {
        "schema": "tinn.section_4_1.s3_production_analysis.v1",
        "evidence_level": "engineering_regression_only",
        "run": str(run),
        "normalization_basis": "100 g initial rasterized binder",
        "binder_mass_g_per_rve": binder_g,
        "qualification": {
            "passed": True,
            "checks": gates,
            "final_time_h": float(qualification["final_time_h"]),
            "accepted_steps": int(qualification["accepted_steps"]),
            "rejected_trials": int(qualification["rejected_trials"]),
            "all_checkpoint_readbacks_match": bool(qualification["all_checkpoint_readbacks_match"]),
            "bundle_sha256_unchanged": bool(qualification["bundle_sha256_unchanged"]),
        },
        "input_sha256": {
            "qualification.json": sha256(run / "qualification.json"),
            "summary.json": sha256(run / "summary.json"),
            "step_audit.jsonl": sha256(run / "step_audit.jsonl"),
        },
        "sulfate_depletion": {
            "initial_primary_carriers_mmol_per_100g": initial_carrier,
            "first_observed_carrier_below_1pct_h": carrier_depletion,
            "first_observed_aqueous_s_below_0p01_mmol_100g_h": aqueous_depletion,
            "resolution_note": "Output-age observation only; not an exact within-step depletion time.",
        },
        "logger_gap": {
            "stepwise_unmet_release_increment": False,
            "trace_water_frozen_cluster_count": False,
            "space_fill_frozen_cluster_count": False,
            "secondary_gypsum_net_flux": False,
            "consequence": "Exact step attribution requires a logger-enhanced rerun; checkpoint analyses do not.",
        },
        "records": records,
        "cluster_centroids": all_centroids,
    }
    (outdir / "s3_production_analysis.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (outdir / "s3_production_timeseries.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_h", "CSHQ_mmol_100g", "Portlandite_mmol_100g", "ettringite_mmol_100g", "Gypsum_mmol_100g", "water_free_g_100g", "water_gel_g_100g", "water_bound_g_100g", "porosity_capillary", "porosity_connected", "CSHQ_bulk_ca_si", "CSHQ_cluster_weighted_ca_si", "aqueous_S_mmol_100g", "primary_sulfate_carrier_mmol_100g"])
        for r in records:
            writer.writerow([r["time_h"], r["phase_mmol_per_100g"].get("CSHQ", 0), r["phase_mmol_per_100g"].get("Portlandite", 0), r["phase_mmol_per_100g"].get("ettringite", 0), r["phase_mmol_per_100g"].get("Gypsum", 0), r["water_g_per_100g"]["free"], r["water_g_per_100g"]["gel"], r["water_g_per_100g"]["bound"], r["porosity_capillary"], r["porosity_split"]["connected"], r["CSHQ_bulk_ca_si"], r["CSHQ_cluster_ca_si"].get("mean_pool_weighted"), r["sulfate"]["aqueous_s_mmol_per_100g"], r["sulfate"]["remaining_primary_carriers_mmol_per_100g"]])
    plot_outputs(records, outdir, map_images, all_centroids)
    (outdir / "S3_PRODUCTION_ANALYSIS_REPORT_KO.md").write_text(report(payload), encoding="utf-8")
    print(json.dumps({"outdir": str(outdir), "checkpoints": len(records), "carrier_depletion_h": carrier_depletion, "aqueous_s_depletion_h": aqueous_depletion}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
