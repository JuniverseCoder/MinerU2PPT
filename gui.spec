# -*- mode: python ; coding: utf-8 -*-

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata

project_root = Path(globals().get("__file__", os.path.abspath("gui.spec"))).resolve().parent
models_src = project_root / "models" / "paddleocr"
datas = []
if models_src.exists():
    datas.append((str(models_src), "models/paddleocr"))

# PaddleOCR 3.x depends on PaddleX pipeline configs (for pipeline="OCR")
# and internal OCR resources at runtime. Bundle these data files explicitly
# to avoid runtime initialization failures in packaged builds.
try:
    datas += collect_data_files(
        "paddlex",
        includes=[
            "configs/**/*.yaml",
            "configs/**/*.yml",
            "configs/**/*.json",
            "**/.version",
            "**/*.ttf",
        ],
    )
except Exception:
    pass

try:
    datas += collect_data_files(
        "paddleocr",
        includes=["**/*.yaml", "**/*.yml", "**/*.json", "**/*.txt"],
    )
except Exception:
    pass

# PaddleX checks optional extras via package metadata at runtime
# (importlib.metadata). In onefile mode this metadata is not always
# automatically collected, which can trigger false DependencyError.
try:
    datas += copy_metadata("paddlex", recursive=True)
except Exception:
    pass

try:
    datas += copy_metadata("paddleocr", recursive=True)
except Exception:
    pass

# PaddleX checks extras at runtime via importlib.metadata and then checks
# dependency package metadata/version. Explicitly include metadata for the
# OCR-core dependency chain to avoid false negatives in onefile mode.
for pkg in [
    "imagesize",
    "opencv-contrib-python",
    "pyclipper",
    "pypdfium2",
    "python-bidi",
    "Shapely",
]:
    try:
        datas += copy_metadata(pkg, recursive=True)
    except Exception:
        pass

binaries = []

# Paddle runtime shared libraries are required in frozen apps.
# onefile extraction may miss these unless they are explicitly collected.
try:
    binaries += collect_dynamic_libs("paddle")
except Exception:
    pass


a = Analysis(
    ['gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MinerU2PPT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MinerU2PPT',
)
