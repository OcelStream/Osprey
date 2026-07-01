import os
import re
from typing import List

from pydantic import Field, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict

# Maps DS_TRACKER env value to the corresponding YAML config file.
# Set DS_TRACKER=off to disable tracking entirely.
_TRACKER_CONFIG_DIR = "/deepstream_app/deepstream/config"
_TRACKER_PRESETS = {
    "IOU":        f"{_TRACKER_CONFIG_DIR}/config_tracker_IOU.yml",
    "NvSORT":     f"{_TRACKER_CONFIG_DIR}/config_tracker_NvSORT.yml",
    "NvDCF":      f"{_TRACKER_CONFIG_DIR}/config_tracker_NvDCF_perf.yml",
    "NvDeepSORT": f"{_TRACKER_CONFIG_DIR}/config_tracker_NvDeepSORT.yml",
}


class PipelineSettings(BaseSettings):
    """Pipeline configuration loaded from environment variables.

    Env vars are read with the ``DS_`` prefix where applicable.
    Legacy names (``batched_push_timeout``, ``WIDTH_MODEL``, ``HEIGHT_MODEL``)
    are also accepted for backward compatibility.

    GIE inference configs are scanned from ``GIE_N_CONFIG`` env vars via a
    property, since they use a dynamic numbering scheme that Pydantic fields
    cannot capture directly.

    Tracker algorithm is selected via ``DS_TRACKER`` env var:
        IOU        — fastest, no GPU, pure bounding-box overlap
        NvSORT     — fast, no GPU, Kalman filter + cascaded matching (default)
        NvDCF      — medium GPU, visual correlation filter, handles occlusion
        NvDeepSORT — high GPU, Re-ID network, best re-identification
        off        — disable tracker entirely
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    batched_push_timeout: int = Field(
        default=66_666,
        validation_alias=AliasChoices("DS_BATCHED_PUSH_TIMEOUT", "batched_push_timeout"),
    )
    model_width: int = Field(
        default=640,
        validation_alias=AliasChoices("DS_MODEL_WIDTH", "WIDTH_MODEL"),
    )
    model_height: int = Field(
        default=640,
        validation_alias=AliasChoices("DS_MODEL_HEIGHT", "HEIGHT_MODEL"),
    )
    tracker: str = Field(
        default="NvSORT",
        validation_alias=AliasChoices("DS_TRACKER", "TRACKER"),
    )
    tracker_width: int = Field(
        default=640,
        validation_alias=AliasChoices("DS_TRACKER_WIDTH", "TRACKER_WIDTH"),
    )
    tracker_height: int = Field(
        default=384,
        validation_alias=AliasChoices("DS_TRACKER_HEIGHT", "TRACKER_HEIGHT"),
    )
    codec: str = "H265"
    bitrate: int = 4_000_000
    meta_serialization_lib: str = (
        "/deepstream_app/deepstream/app/lib/serialize_meta.so"
    )
    perf_interval_ms: int = 5_000

    @property
    def tracker_ll_lib(self) -> str:
        return "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"

    @property
    def tracker_config(self) -> str:
        """Resolve DS_TRACKER name to YAML config path. Empty string disables tracker."""
        name = self.tracker.strip()
        if not name or name.lower() == "off":
            return ""
        if name in _TRACKER_PRESETS:
            return _TRACKER_PRESETS[name]
        # Allow a raw file path for custom configs
        return name

    @property
    def gie_configs(self) -> List[str]:
        """Return ordered list of GIE config paths from GIE_N_CONFIG env vars."""
        gie_map: dict = {}
        for key, value in os.environ.items():
            m = re.match(r"^GIE_(\d+)_CONFIG$", key)
            if m:
                gie_map[int(m.group(1))] = value.strip()
        return [gie_map[i] for i in sorted(gie_map)]


settings = PipelineSettings()
