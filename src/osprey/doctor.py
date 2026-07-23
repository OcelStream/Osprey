"""osprey-doctor — verify the Osprey runtime is complete in THIS interpreter.

Osprey needs three pieces in the *same* Python interpreter:
  - osprey  (this package, from pip)
  - gi      (PyGObject — the system apt package python3-gi)
  - pyds    (DeepStream bindings — built by osprey-bootstrap)

They come from different places (pip vs apt vs a native build), so the common
failure is having them land in *different* interpreters (e.g. ospreyai in a
venv, gi in system Python). This tool checks the current interpreter and prints
the exact fix for whatever is missing.

Run it with the SAME python you use to run your app:

    osprey-doctor
    # or:  python3 -m osprey.doctor
"""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys


def _try_import(mod: str):
    try:
        m = importlib.import_module(mod)
        return True, getattr(m, "__file__", None)
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> int:
    print(f"osprey-doctor — interpreter: {sys.executable}\n")
    ok = True

    # --- osprey (this package) ---
    good, info = _try_import("osprey")
    print(f"[{'OK  ' if good else 'FAIL'}] osprey        {info or ''}")
    if not good:
        print("       -> pip install ospreyai (into THIS interpreter)")
        ok = False

    # --- gi (system apt package) ---
    good, info = _try_import("gi")
    print(f"[{'OK  ' if good else 'FAIL'}] gi (PyGObject) {info or ''}")
    if not good:
        print("       -> 'gi' is the system apt package python3-gi; a plain venv cannot see it.")
        print("          Fix: run with system python3, OR recreate the venv with")
        print("               'python3 -m venv --system-site-packages <venv>'.")
        ok = False

    # --- pyds (must be a REAL compiled module, not a namespace phantom) ---
    good, info = _try_import("pyds")
    is_real = good and isinstance(info, str) and info.endswith(".so")
    if is_real:
        print(f"[OK  ] pyds          {info}")
    elif good:
        print(f"[FAIL] pyds          imported but not a compiled module ({info!r})")
        print("       -> a stray namespace 'pyds' is masking a missing build.")
        print("          Fix: OSPREY_ONLY=30 osprey-bootstrap   (run with THIS python active)")
        ok = False
    else:
        print(f"[FAIL] pyds          {info}")
        print("       -> not built for this interpreter.")
        print("          Fix: OSPREY_ONLY=30 osprey-bootstrap   (run with THIS python active)")
        ok = False

    # --- DeepStream GStreamer plugin ---
    if shutil.which("gst-inspect-1.0"):
        rc = subprocess.run(
            ["gst-inspect-1.0", "nvstreammux"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode
        print(f"[{'OK  ' if rc == 0 else 'FAIL'}] nvstreammux    "
              f"{'available' if rc == 0 else 'missing — run: sudo osprey-bootstrap'}")
        ok = ok and rc == 0
    else:
        print("[WARN] gst-inspect-1.0 not found — GStreamer tools missing "
              "(run: sudo osprey-bootstrap)")

    print()
    if ok:
        print("All good — gi + pyds + osprey + DeepStream plugins share this interpreter.")
        return 0
    print("Problems found (see fixes above). Osprey needs gi + pyds + osprey in ONE interpreter.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
