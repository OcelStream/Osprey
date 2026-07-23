"""Packaging shim — forces a platform-tagged wheel.

All project metadata lives in ``pyproject.toml``. This file exists only to mark
the distribution as *impure* so the built wheel carries a platform tag
(e.g. ``osprey-0.1.0-cp310-cp310-manylinux_2_39_x86_64.whl``) instead of the
default ``py3-none-any``.

Osprey ships precompiled native libraries (``osprey/**/lib/*.so``) built for
DeepStream 8.0 / CUDA 12.8 on x86_64. A ``py3-none-any`` wheel would install on
incompatible platforms (ARM, macOS, other CUDA/DS versions) and then crash at
import. The platform tag makes pip install this wheel only where its ``.so``
actually work, and fall back to the sdist elsewhere.

Build ON the target platform (Ubuntu 24.04 / DeepStream host or container) so
the manylinux/x86_64 tag matches the machine that will run it.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Distribution that always reports as having extension modules.

    ``has_ext_modules() -> True`` is what setuptools/bdist_wheel checks to
    decide whether the wheel is platform-specific. We have no compiled
    ext_modules (the ``.so`` ship as package data), so we assert it here.
    """

    def has_ext_modules(self) -> bool:  # noqa: D401
        return True


setup(distclass=BinaryDistribution)
