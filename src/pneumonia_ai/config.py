"""Dataset path configuration loaded from environment variables."""

from dataclasses import dataclass
from os import environ
from pathlib import Path


@dataclass(frozen=True)
class DatasetPaths:
    """Filesystem locations for the supported datasets."""

    chexpert_root: Path
    mimic_cxr_root: Path
    nih_cxr_root: Path

    @classmethod
    def from_environment(cls) -> "DatasetPaths":
        """Build dataset paths from required environment variables.

        Raises:
            EnvironmentError: If one or more required variables are unset or empty.
        """
        variable_names = (
            "CHEXPERT_ROOT",
            "MIMIC_CXR_ROOT",
            "NIH_CXR_ROOT",
        )
        missing_variables = [
            variable_name
            for variable_name in variable_names
            if not environ.get(variable_name, "").strip()
        ]
        if missing_variables:
            missing = ", ".join(missing_variables)
            raise EnvironmentError(
                f"Missing required dataset environment variable(s): {missing}. "
                "Set them before loading dataset configuration."
            )

        return cls(
            chexpert_root=Path(environ["CHEXPERT_ROOT"]).expanduser(),
            mimic_cxr_root=Path(environ["MIMIC_CXR_ROOT"]).expanduser(),
            nih_cxr_root=Path(environ["NIH_CXR_ROOT"]).expanduser(),
        )

    def existence_report(self) -> dict[str, bool]:
        """Return whether each configured dataset root currently exists."""
        return {
            "CheXpert": self.chexpert_root.exists(),
            "MIMIC-CXR": self.mimic_cxr_root.exists(),
            "NIH ChestX-ray14": self.nih_cxr_root.exists(),
        }
