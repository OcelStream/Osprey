import os
import re
from pathlib import Path
from typing import List, Optional

from pydantic import Field, AliasChoices, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

from osprey.paths import default_socket_dir

# ---------------------------------------------------------------------------
# Packaged resource locations.
#
# Osprey ships its tracker configs and native meta-serialization library as
# package data, so defaults resolve relative to the installed package rather
# than an assumed /deepstream_app bind-mount. Every path below can still be
# overridden via environment variables.
# ---------------------------------------------------------------------------
_SERVER_DIR = Path(__file__).resolve().parent.parent          # osprey/server
_CONFIG_DIR = _SERVER_DIR / "config"
_LIB_DIR = _SERVER_DIR / "deepstream" / "lib"

# Maps DS_TRACKER env value to the corresponding packaged YAML config file.
# Set DS_TRACKER=off to disable tracking entirely.
_TRACKER_PRESETS = {
    "IOU":        str(_CONFIG_DIR / "config_tracker_IOU.yml"),
    "NvSORT":     str(_CONFIG_DIR / "config_tracker_NvSORT.yml"),
    "NvDCF":      str(_CONFIG_DIR / "config_tracker_NvDCF_perf.yml"),
    "NvDeepSORT": str(_CONFIG_DIR / "config_tracker_NvDeepSORT.yml"),
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
    # Directory holding the per-stream IPC sockets. Defaults to ./sockets in
    # the working directory the app was launched from — the client watches the
    # same path (see osprey.paths).
    socket_dir: str = Field(
        default_factory=default_socket_dir,
        validation_alias=AliasChoices("OSPREY_SOCKET_DIR", "DS_SOCKET_DIR", "socket_dir"),
    )
    meta_serialization_lib: str = Field(
        default=str(_LIB_DIR / "serialize_meta.so"),
        validation_alias=AliasChoices("DS_META_SERIALIZATION_LIB", "meta_serialization_lib"),
    )
    perf_interval_ms: int = 5_000

    # Programmatic override for the GIE (inference) configs. When set via
    # ``with_gie_configs`` / ``configure``, it takes precedence over the
    # ``GIE_N_CONFIG`` environment scan below.
    _gie_configs_override: Optional[List[str]] = PrivateAttr(default=None)

    def with_gie_configs(self, paths: List[str]) -> "PipelineSettings":
        """Set the GIE config paths programmatically (overrides GIE_N_CONFIG env)."""
        self._gie_configs_override = [str(p) for p in paths]
        return self

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
        """Return ordered list of GIE config paths.

        A programmatic override (``with_gie_configs``) wins; otherwise the
        paths are scanned from ``GIE_N_CONFIG`` environment variables.
        """
        if self._gie_configs_override is not None:
            return self._gie_configs_override
        gie_map: dict = {}
        for key, value in os.environ.items():
            m = re.match(r"^GIE_(\d+)_CONFIG$", key)
            if m:
                gie_map[int(m.group(1))] = value.strip()
        return [gie_map[i] for i in sorted(gie_map)]


settings = PipelineSettings()
