from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from sae_seed_similarity.plotting import (
    _paper_assignment_class,
    paper_encoder_decoder_joint,
    paper_hungarian_vs_max_cosine,
    paper_threshold_sweep,
    save_figure,
)


def _paper_matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decoder_matched_cosine": [0.90, 0.65, 0.40],
            "encoder_matched_cosine": [0.92, 0.64, 0.42],
            "decoder_max_cosine": [0.90, 0.70, 0.55],
            "encoder_max_cosine": [0.92, 0.72, 0.54],
            # The second row verifies that assignment equality is independent
            # of the stricter shared/orphan classification.
            "same_counterpart": [True, True, False],
            "is_shared": [True, False, False],
        }
    )


def test_paper_assignment_class_uses_counterpart_equality() -> None:
    values = _paper_assignment_class(_paper_matches())
    assert values.tolist() == ["Equal", "Equal", "Different"]


def test_encoder_decoder_joint_uses_equal_different_classification() -> None:
    figure = paper_encoder_decoder_joint(_paper_matches(), threshold=0.7)
    legend = next(axis.get_legend() for axis in figure.axes if axis.get_legend())
    assert legend.get_title().get_text() == "Encoder-decoder\nHungarian matching"
    assert [text.get_text() for text in legend.get_texts()] == ["Equal", "Different"]
    joint_axis = next(axis for axis in figure.axes if axis.get_legend() is legend)
    assert all(line.get_linestyle() != "--" for line in joint_axis.lines)
    plt.close(figure)


def test_hungarian_vs_max_uses_equal_different_legend_without_title() -> None:
    figure = paper_hungarian_vs_max_cosine(_paper_matches(), direction="decoder")
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert legend.get_title().get_text() == ""
    assert [text.get_text() for text in legend.get_texts()] == ["Equal", "Different"]
    plt.close(figure)


def test_threshold_sweep_uses_cleaned_labels() -> None:
    frame = pd.DataFrame(
        {
            "threshold": [0.0, 0.7, 1.0],
            "cosine_threshold_fraction": [1.0, 0.5, 0.0],
            "shared_fraction": [0.5, 0.4, 0.0],
            "max_cosine_fraction": [1.0, 0.6, 0.0],
        }
    )
    figure = paper_threshold_sweep(frame, selected_threshold=0.7)
    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == [
        "Threshold cosine similarity",
        "Equal + Threshold cosine similarity",
        "Max cosine similarity",
        "Selected threshold = 0.7",
    ]
    plt.close(figure)


def test_save_figure_removes_grid_lines(tmp_path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    axis.grid(True)
    save_figure(figure, tmp_path, "without_grid")
    assert not any(line.get_visible() for line in axis.get_xgridlines())
    assert not any(line.get_visible() for line in axis.get_ygridlines())
