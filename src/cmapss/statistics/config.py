"""Cấu hình và trạng thái dùng chung cho suy luận sensor FD001."""

from dataclasses import dataclass

from cmapss.data.contracts import ContractError, SENSORS

_BOOTSTRAP_PURPOSE = 1
_WILCOXON_PURPOSE = 2
_TEST_STATUS_OK = "ok"
_TEST_STATUS_ALL_ZERO = "all_zero_difference"
_BOOTSTRAP_STATUS_OK = "ok"
_BOOTSTRAP_STATUS_DEGENERATE = "degenerate"


@dataclass(frozen=True)
class SensorInferenceConfig:
    """Cấu hình suy luận trên differences theo sensor."""

    bootstrap_resamples: int = 10_000
    bootstrap_confidence_level: float = 0.95
    bootstrap_method: str = "percentile"
    bootstrap_quantile_method: str = "linear"
    wilcoxon_alternative: str = "two-sided"
    wilcoxon_zero_method: str = "wilcox"
    wilcoxon_correction: bool = False
    wilcoxon_method: str = "permutation"
    wilcoxon_permutation_resamples: int = 100_000
    fdr_method: str = "bh"
    fdr_q: float = 0.05
    fdr_family_size: int = len(SENSORS)
    seed: int = 42

    def validate(self) -> None:
        """Chặn cấu hình khác với phương pháp đã cố định."""
        if self.bootstrap_method != "percentile":
            raise ContractError("INFERENCE_CONFIG", "bootstrap_method", "chỉ hỗ trợ percentile")
        if self.bootstrap_resamples < 1:
            raise ContractError("INFERENCE_CONFIG", "bootstrap_resamples", "cần số lần bootstrap dương")
        if not 0.0 < self.bootstrap_confidence_level < 1.0:
            raise ContractError("INFERENCE_CONFIG", "bootstrap_confidence_level", "cần nằm trong (0, 1)")
        if self.wilcoxon_alternative != "two-sided":
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_alternative", "chỉ hỗ trợ two-sided")
        if self.wilcoxon_zero_method != "wilcox":
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_zero_method", "chỉ hỗ trợ wilcox")
        if self.wilcoxon_correction:
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_correction", "không dùng continuity correction")
        if self.wilcoxon_method != "permutation":
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_method", "cần permutation, không dùng auto")
        if self.wilcoxon_permutation_resamples < 1:
            raise ContractError("INFERENCE_CONFIG", "wilcoxon_permutation_resamples", "cần số lần permutation dương")
        if self.fdr_method != "bh":
            raise ContractError("INFERENCE_CONFIG", "fdr_method", "chỉ hỗ trợ Benjamini-Hochberg")
        if not 0.0 < self.fdr_q < 1.0:
            raise ContractError("INFERENCE_CONFIG", "fdr_q", "cần nằm trong (0, 1)")
        if self.fdr_family_size != len(SENSORS):
            raise ContractError("INFERENCE_CONFIG", "fdr_family_size", "family phải gồm đủ 21 sensors")
