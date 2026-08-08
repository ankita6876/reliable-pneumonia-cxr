import json
import zipfile

from scripts.analysis.freeze_thesis_baseline import (
    _read_zip_json,
    _t_two_sided_p_df2,
    _wilcoxon_exact_two_sided,
)


def test_df2_two_sided_t_p_value_is_symmetric() -> None:
    assert _t_two_sided_p_df2(0.0) == 1.0
    assert _t_two_sided_p_df2(2.0) == _t_two_sided_p_df2(-2.0)


def test_three_seed_wilcoxon_exact_values() -> None:
    assert _wilcoxon_exact_two_sided([-0.1, -0.2, -0.3]) == 0.25
    assert _wilcoxon_exact_two_sided([0.3, -0.2, -0.1]) == 1.0


def test_archive_json_parser_reads_a_single_matching_member(tmp_path) -> None:
    archive = tmp_path / "baseline.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("run/validation_metrics.json", json.dumps({"auroc": 0.75}))
    assert _read_zip_json(archive, "validation_metrics.json") == {"auroc": 0.75}
