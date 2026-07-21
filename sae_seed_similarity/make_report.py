"""Create reproducible summary tables, statistical controls, plots, and report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .config import EvaluationConfig, load_config
from .plotting import (
    distribution,
    heatmap,
    paper_encoder_decoder_joint,
    paper_hungarian_vs_max_cosine,
    paper_shared_orphan_distribution,
    paper_shared_orphan_fractions,
    paper_similarity_vs_frequency,
    paper_threshold_sweep,
    save_figure as _save_figure,
    scatter,
)
from .statistics import bootstrap_ci, matched_control_statistics
from .storage import ArtifactStore
from .utils import configure_logging

LOGGER = logging.getLogger(__name__)


def _read_if(path: Path, parquet: bool = False) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if parquet else pd.read_csv(path)


def _control_summary(config: EvaluationConfig, store: ArtifactStore) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    global_controls = _read_if(store.root / "controls_summary.csv")
    if len(global_controls):
        rows.extend(global_controls.to_dict("records"))
    overlap_controls = _read_if(store.root / "control_activation_overlap.parquet", True)
    if len(overlap_controls):
        for (sae_a, sae_b, control), frame in overlap_controls.groupby(
            ["sae_a", "sae_b", "control"]
        ):
            interval = bootstrap_ci(
                frame["jaccard"].to_numpy(),
                samples=config.bootstrap.samples,
                confidence_level=config.bootstrap.confidence_level,
                random_seed=config.dataset.random_seed,
            )
            rows.append(
                {
                    "sae_a": sae_a,
                    "sae_b": sae_b,
                    "control": control,
                    "metric": "activation_jaccard",
                    "estimate": interval.estimate,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(store.root / "controls_summary.csv", index=False)
    return result


def _statistical_summary(
    config: EvaluationConfig, store: ArtifactStore
) -> pd.DataFrame:
    """Summarize token, feature-pair, and seed-pair uncertainty and controls."""
    rows: list[dict[str, Any]] = []
    random_pairs = _read_if(store.root / "random_feature_pairs.parquet", True)
    matched_overlap = _read_if(store.root / "activation_overlap.parquet", True)
    control_overlap = _read_if(store.root / "control_activation_overlap.parquet", True)
    if len(random_pairs) and len(matched_overlap) and len(control_overlap):
        random_overlap = control_overlap[control_overlap["control"] == "random_pair"]
        mapped = (
            random_overlap
            if "matched_feature_b" in random_overlap
            else random_overlap.merge(
                random_pairs,
                on=["sae_a", "sae_b", "feature_a", "feature_b"],
            )
        )
        random_by_match = (
            mapped.groupby(["sae_a", "sae_b", "feature_a", "matched_feature_b"])[
                "jaccard"
            ]
            .mean()
            .rename("control_value")
        )
        matched = matched_overlap.rename(
            columns={"feature_b": "matched_feature_b"}
        ).set_index(["sae_a", "sae_b", "feature_a", "matched_feature_b"])
        joined = matched.join(random_by_match, how="inner")
        if len(joined):
            stats = matched_control_statistics(
                joined["jaccard"].to_numpy(),
                joined["control_value"].to_numpy(),
                random_seed=config.dataset.random_seed,
            )
            rows.append(
                {"level": "feature_pair", "metric": "activation_jaccard", **stats}
            )

    seed_pairs = _read_if(store.root / "seed_pair_summary.csv")
    if len(seed_pairs) >= 3:
        for metric in ("cka", "cka_standardized", "svcca_mean"):
            interval = bootstrap_ci(
                seed_pairs[metric].to_numpy(),
                samples=config.bootstrap.samples,
                confidence_level=config.bootstrap.confidence_level,
                random_seed=config.dataset.random_seed,
            )
            rows.append(
                {
                    "level": "seed_pair",
                    "metric": metric,
                    "estimate": interval.estimate,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(store.root / "statistical_summary.csv", index=False)
    return result


def _make_plots(config: EvaluationConfig, store: ArtifactStore) -> None:
    plot_dir = store.root / "plots"
    plot_progress = tqdm(desc="report plots", unit="plot")

    def save_figure(figure: Any, output_dir: Path, name: str) -> None:
        plot_progress.set_postfix_str(name, refresh=True)
        _save_figure(figure, output_dir, name)
        plot_progress.update()

    for filename, title, output_name in (
        ("cka_matrix.csv", "Linear CKA across SAE seeds", "cka_heatmap"),
        ("svcca_matrix.csv", "Mean SVCCA across SAE seeds", "svcca_heatmap"),
    ):
        path = store.root / filename
        if path.exists():
            matrix = pd.read_csv(path, index_col=0)
            save_figure(heatmap(matrix, title, "SAE seed"), plot_dir, output_name)

    overlap = _read_if(store.root / "activation_overlap.parquet", True)
    control_overlap = _read_if(store.root / "control_activation_overlap.parquet", True)
    if len(overlap):
        save_figure(
            scatter(
                overlap,
                "decoder_cosine",
                "jaccard",
                title="Decoder similarity vs activation overlap",
            ),
            plot_dir,
            "decoder_cosine_vs_activation_jaccard",
        )
    if len(overlap) and len(control_overlap):
        matched = overlap[["jaccard"]].copy()
        matched["pair_type"] = "matched"
        random_values = control_overlap[control_overlap["control"] == "random_pair"][
            ["jaccard"]
        ].copy()
        random_values["pair_type"] = "random"
        frame = pd.concat([matched, random_values], ignore_index=True)
        save_figure(
            distribution(
                frame,
                "jaccard",
                "pair_type",
                title="Matched vs random activation overlap",
            ),
            plot_dir,
            "matched_vs_random_distributions",
        )

    paper_matches = _read_if(store.root / "paper_hungarian_matches.parquet", True)
    paper_summary = _read_if(store.root / "paper_seed_pair_summary.csv")
    threshold_sweep = _read_if(store.root / "paper_threshold_sweep.csv")
    if len(paper_matches):
        pair_count = paper_matches.groupby(["sae_a", "sae_b"]).ngroups
        for (sae_a, sae_b), frame in paper_matches.groupby(
            ["sae_a", "sae_b"], sort=False
        ):
            suffix = "" if pair_count == 1 else f"_{sae_a}__{sae_b}"
            save_figure(
                paper_encoder_decoder_joint(
                    frame, threshold=config.paper_matching.shared_threshold
                ),
                plot_dir,
                f"paper_figure_1_encoder_decoder_alignment{suffix}",
            )
            save_figure(
                paper_hungarian_vs_max_cosine(frame, direction="decoder"),
                plot_dir,
                f"paper_figure_a2_decoder_hungarian_vs_max_cosine{suffix}",
            )
            save_figure(
                paper_hungarian_vs_max_cosine(frame, direction="encoder"),
                plot_dir,
                f"paper_encoder_hungarian_vs_max_cosine{suffix}",
            )
            save_figure(
                paper_shared_orphan_distribution(frame),
                plot_dir,
                f"paper_shared_orphan_cosine_distributions{suffix}",
            )
            if len(overlap):
                frequencies = overlap[
                    (overlap["sae_a"] == sae_a) & (overlap["sae_b"] == sae_b)
                ][["feature_a", "activation_frequency_a"]]
                frequency_frame = frame.merge(
                    frequencies.drop_duplicates("feature_a"),
                    on="feature_a",
                    how="inner",
                )
                if len(frequency_frame):
                    save_figure(
                        paper_similarity_vs_frequency(frequency_frame),
                        plot_dir,
                        f"paper_similarity_vs_firing_frequency{suffix}",
                    )
    if len(threshold_sweep):
        pair_count = threshold_sweep.groupby(["sae_a", "sae_b"]).ngroups
        for (sae_a, sae_b), frame in threshold_sweep.groupby(
            ["sae_a", "sae_b"], sort=False
        ):
            suffix = "" if pair_count == 1 else f"_{sae_a}__{sae_b}"
            save_figure(
                paper_threshold_sweep(
                    frame,
                    selected_threshold=config.paper_matching.shared_threshold,
                ),
                plot_dir,
                f"paper_figure_a1_threshold_sweep{suffix}",
            )
    if len(paper_summary):
        save_figure(
            paper_shared_orphan_fractions(paper_summary),
            plot_dir,
            "paper_shared_orphan_fractions",
        )
    spectra_files = sorted((store.root / "svcca_correlations").glob("*.npz"))
    if spectra_files:
        figure, axis = plt.subplots(figsize=(8, 5))
        pca_figure, pca_axis = plt.subplots(figsize=(8, 5))
        for path in spectra_files:
            values = np.load(path)
            axis.plot(values["correlations"])
            pca_axis.plot(values["pca_curve_a"], alpha=0.7, label="A")
            pca_axis.plot(
                values["pca_curve_b"],
                alpha=0.7,
                linestyle="--",
                label="B",
            )
        axis.set(
            title="Canonical-correlation spectra",
            xlabel="Canonical component",
            ylabel="Correlation",
            ylim=(0, 1.02),
        )
        pca_axis.set(
            title="PCA explained-variance curves",
            xlabel="Principal component",
            ylabel="Cumulative explained variance",
            ylim=(0, 1.02),
        )
        pca_axis.legend(fontsize="small")
        save_figure(figure, plot_dir, "canonical_correlation_spectra")
        save_figure(pca_figure, plot_dir, "pca_explained_variance_curves")
    plot_progress.close()


def run(config: EvaluationConfig) -> Path:
    store = ArtifactStore(config.output_path).ensure()
    controls = _control_summary(config, store)
    statistics = _statistical_summary(config, store)
    _make_plots(config, store)
    summary = _read_if(store.root / "seed_pair_summary.csv")
    paper_summary = _read_if(store.root / "paper_seed_pair_summary.csv")
    report_path = store.root / "report.md"
    lines = [
        "# SAE seed similarity report",
        "",
        "Generated from the exact configuration captured in `run_manifest.json`.",
        "",
        "## Seed-pair representation summary",
        "",
        summary.to_markdown(index=False)
        if len(summary)
        else "Representation stage not run.",
        "",
        "## Paper shared/orphan summary",
        "",
        paper_summary.to_markdown(index=False)
        if len(paper_summary)
        else "Paper matching stage not run.",
        "",
        "## Controls",
        "",
        controls.to_markdown(index=False)
        if len(controls)
        else "Control stages not run.",
        "",
        "## Statistical comparisons",
        "",
        statistics.to_markdown(index=False)
        if len(statistics)
        else "Insufficient completed controls for statistical comparisons.",
        "",
        "## Interpretation guardrail",
        "",
        "Feature matching, activation overlap, CKA, and SVCCA establish correlational or geometric similarity; they do not establish causal equivalence.",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote report and plots under %s", store.root)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
