"""Create reproducible summary tables, statistical controls, plots, and report."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import EvaluationConfig, load_config
from .plotting import distribution, heatmap, save_figure, scatter
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
    feature_ablation = _read_if(store.root / "ablation_feature_level.parquet", True)
    if len(feature_ablation) and "pair_type" in feature_ablation:
        for pair_type, frame in feature_ablation.groupby("pair_type"):
            for metric in (
                "mean_logit_delta_cosine",
                "mean_ablation_jsd_between_seeds",
                "top1_disagreement_rate",
            ):
                if metric in frame:
                    interval = bootstrap_ci(
                        frame[metric].to_numpy(),
                        samples=config.bootstrap.samples,
                        confidence_level=config.bootstrap.confidence_level,
                        random_seed=config.dataset.random_seed,
                    )
                    rows.append(
                        {
                            "control": pair_type,
                            "metric": metric,
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

    feature_ablation = _read_if(store.root / "ablation_feature_level.parquet", True)
    if len(feature_ablation) and len(random_pairs):
        matched = feature_ablation[feature_ablation["pair_type"] == "matched"].rename(
            columns={"feature_b": "matched_feature_b"}
        )
        random_values = feature_ablation[
            feature_ablation["pair_type"] == "random_control"
        ].merge(
            random_pairs,
            on=["sae_a", "sae_b", "feature_a", "feature_b"],
        )
        keys = ["sae_a", "sae_b", "feature_a", "matched_feature_b"]
        for metric in (
            "mean_logit_delta_cosine",
            "mean_ablation_jsd_between_seeds",
            "top1_disagreement_rate",
        ):
            if metric not in matched or metric not in random_values:
                continue
            control_mean = (
                random_values.groupby(keys)[metric].mean().rename("control_value")
            )
            joined = matched.set_index(keys).join(control_mean, how="inner")
            stats = matched_control_statistics(
                joined[metric].to_numpy(),
                joined["control_value"].to_numpy(),
                random_seed=config.dataset.random_seed,
            )
            rows.append({"level": "feature_pair", "metric": metric, **stats})

    seed_pairs = _read_if(store.root / "seed_pair_summary.csv")
    if len(seed_pairs) >= 3:
        for metric in ("cka", "cka_standardized", "svcca_mean", "pwcca"):
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


def _make_plots(store: ArtifactStore) -> None:
    plot_dir = store.root / "plots"
    for filename, title, output_name in (
        ("cka_matrix.csv", "Linear CKA across SAE seeds", "cka_heatmap"),
        ("svcca_matrix.csv", "Mean SVCCA across SAE seeds", "svcca_heatmap"),
        ("pwcca_matrix.csv", "PWCCA across SAE seeds", "pwcca_heatmap"),
    ):
        path = store.root / filename
        if path.exists():
            matrix = pd.read_csv(path, index_col=0)
            save_figure(heatmap(matrix, title, "SAE seed"), plot_dir, output_name)

    overlap = _read_if(store.root / "activation_overlap.parquet", True)
    ablation = _read_if(store.root / "ablation_feature_level.parquet", True)
    prompt = _read_if(store.root / "ablation_prompt_level.parquet", True)
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
    merged = pd.DataFrame()
    if len(overlap) and len(ablation):
        keys = ["sae_a", "sae_b", "feature_a", "feature_b"]
        merged = overlap.merge(
            ablation[ablation.get("pair_type", "matched") == "matched"], on=keys
        )
        if len(merged):
            save_figure(
                scatter(
                    merged,
                    "decoder_cosine",
                    "mean_logit_delta_cosine",
                    title="Decoder similarity vs causal-effect similarity",
                ),
                plot_dir,
                "decoder_cosine_vs_logit_delta_cosine",
            )
            save_figure(
                scatter(
                    merged,
                    "jaccard",
                    "mean_logit_delta_cosine",
                    title="Activation overlap vs functional similarity",
                ),
                plot_dir,
                "activation_overlap_vs_functional_similarity",
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
    if len(prompt):
        effect = (prompt["effect_jsd_a"] + prompt["effect_jsd_b"]) / 2
        prompt = prompt.assign(mean_effect_jsd=effect)
        save_figure(
            scatter(
                prompt,
                "mean_effect_jsd",
                "ablation_jsd_between_seeds",
                title="Ablation divergence vs effect magnitude",
                hue="pair_type",
            ),
            plot_dir,
            "ablation_jsd_vs_effect_magnitude",
        )
        save_figure(
            distribution(
                prompt,
                "top1_disagreement",
                "pair_type",
                title="Top-1 disagreement by pair type",
            ),
            plot_dir,
            "top1_disagreement_distributions",
        )

    spectra_files = sorted((store.root / "svcca_correlations").glob("*.npz"))
    if spectra_files:
        figure, axis = plt.subplots(figsize=(8, 5))
        pca_figure, pca_axis = plt.subplots(figsize=(8, 5))
        for path in spectra_files:
            values = np.load(path)
            axis.plot(values["correlations"], label=path.stem)
            pca_axis.plot(values["pca_curve_a"], alpha=0.7, label=f"{path.stem}: A")
            pca_axis.plot(
                values["pca_curve_b"],
                alpha=0.7,
                linestyle="--",
                label=f"{path.stem}: B",
            )
        axis.set(
            title="Canonical-correlation spectra",
            xlabel="Canonical component",
            ylabel="Correlation",
            ylim=(0, 1.02),
        )
        axis.legend(fontsize="small")
        pca_axis.set(
            title="PCA explained-variance curves",
            xlabel="Principal component",
            ylabel="Cumulative explained variance",
            ylim=(0, 1.02),
        )
        pca_axis.legend(fontsize="small")
        save_figure(figure, plot_dir, "canonical_correlation_spectra")
        save_figure(pca_figure, plot_dir, "pca_explained_variance_curves")
    if len(merged):
        merged = merged.copy()
        frequency = merged[["activation_frequency_a", "activation_frequency_b"]].mean(
            axis=1
        )
        if frequency.nunique() > 1:
            merged["frequency_bin"] = pd.qcut(
                frequency,
                q=min(10, max(2, len(merged) // 20)),
                duplicates="drop",
            ).astype(str)
            figure, axis = plt.subplots(figsize=(10, 5))
            sns.boxplot(
                data=merged, x="frequency_bin", y="mean_logit_delta_cosine", ax=axis
            )
            axis.tick_params(axis="x", rotation=45)
            axis.set_title("Functional similarity stratified by feature frequency")
            save_figure(figure, plot_dir, "feature_frequency_stratified_results")


def run(config: EvaluationConfig) -> Path:
    store = ArtifactStore(config.output_path).ensure()
    controls = _control_summary(config, store)
    statistics = _statistical_summary(config, store)
    _make_plots(store)
    summary = _read_if(store.root / "seed_pair_summary.csv")
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
        "A pair is not treated as functionally equivalent when both ablations have negligible effects, even if their output distributions are similar.",
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
