# Changelog

All notable changes to `ospreyai` are documented here. Versions follow
[Semantic Versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

## [Unreleased]

## [0.1.5] - 2026-07-23
### Changed
- IPC sockets no longer default to `/run/nvunixfd` (root-owned, container-only).
  The server's `nvunixfdsink` and the client's watcher now share `./sockets`
  under the working directory the app is launched from — created automatically,
  no `sudo` needed. Override with the `OSPREY_SOCKET_DIR` env var,
  `osprey.configure(socket_dir=…)`, or `DeepStreamClient(watch_dir=…)`; set
  `OSPREY_SOCKET_DIR=/run/nvunixfd` to keep the old location.

### Added
- `osprey.paths` helper (`default_socket_dir` / `ensure_socket_dir`) — single
  source of truth both halves read to resolve the socket directory.
- `socket_dir` field on `PipelineSettings` (aliases `OSPREY_SOCKET_DIR` /
  `DS_SOCKET_DIR`).

## [0.1.4] - 2026-07-23
### Added
- `osprey-doctor` CLI — checks that `gi` + `pyds` + `osprey` + the DeepStream
  plugins all import in the current interpreter, and prints the exact fix for
  whatever is missing. The bootstrap runs it automatically at the end.
- README "Python environment" section explaining that `gi`/`pyds` are system
  packages a plain virtualenv can't see, with the `--break-system-packages` and
  `--system-site-packages` install recipes.
- `examples/gie.txt` (documented GIE config template) and
  `examples/make_gie_config.py` (generates a config with the shipped parser
  `.so` path resolved automatically).

### Fixed
- Bootstrap stage 30 no longer false-skips the `pyds` build: it now requires
  `pyds.__file__` to be a real compiled `.so`, so a stray namespace-only `pyds`
  on the path can't leave the host with no working bindings.

## [0.1.3] - 2026-07-23
- Published build (superseded by 0.1.4).

## [0.1.2] - 2026-07-22
### Fixed
- Bootstrap completion message now says `pip install ospreyai` (was the stale
  `pip install "osprey[server]"`).

## [0.1.1] - 2026-07-22
### Changed
- **Single `pip install ospreyai`** installs everything — server deps
  (FastAPI/uvicorn/pydantic) merged into the base dependencies; the `[server]`
  extra is removed. Server and client run on the same host (host-local Unix
  sockets), so both halves are always needed together.
- README: documented the single-file `configure → serve → client` flow as the
  primary usage; removed Docker framing (bootstrap installs DeepStream 8.0
  bare-metal on the host).

### Added
- Programmatic server API: `osprey.configure()`, `osprey.serve()` (forks the
  control plane + pipeline into its own process), `osprey.add_stream()` /
  `remove_stream()` / `list_streams()`, and `osprey.start()` (blocking).
- `add_stream()` guards against a duplicate `stream_id` — logs a warning and
  skips instead of crashing the caller.
- Client: probe each IPC socket for a live server at most once (cached), so the
  initial scan and the watcher stop double-connecting and spamming the server's
  `nvunixfdsink` with "Broken pipe".
- `examples/vehicle_counter.py` — runnable single-file app.
- Packaging: `setup.py` shim forces a platform-tagged wheel; `PUBLISHING.md`
  release runbook.

### Fixed
- Bootstrap stage 00 installs `libmosquitto1` — the `nvtracker` low-level lib
  needs it, and without it the pipeline never leaves READY and every `/add` was
  (misleadingly) rejected with "Pipeline is not running".
- `_require_playing()` now reports the actual pipeline state on failure instead
  of the misleading "call start() first".

## [0.1.0] - 2026-07-22
### Added
- Initial release: dynamic multi-stream DeepStream 8.0 pipeline, FastAPI control
  plane, IPC-over-Unix-socket client, RTSP output, bare-metal bootstrap.

> **Note:** 0.1.0 shipped a `[server]` optional-dependency extra
> (`pip install "ospreyai[server]"`). It is superseded by 0.1.1, where a plain
> `pip install ospreyai` installs everything. 0.1.0 is yanked on PyPI.

[Unreleased]: https://github.com/ILKAY-BRAHIM/osprey/compare/v0.1.4...HEAD
[0.1.4]: https://github.com/ILKAY-BRAHIM/osprey/releases/tag/v0.1.4
[0.1.3]: https://github.com/ILKAY-BRAHIM/osprey/releases/tag/v0.1.3
[0.1.2]: https://github.com/ILKAY-BRAHIM/osprey/releases/tag/v0.1.2
[0.1.1]: https://github.com/ILKAY-BRAHIM/osprey/releases/tag/v0.1.1
[0.1.0]: https://github.com/ILKAY-BRAHIM/osprey/releases/tag/v0.1.0
