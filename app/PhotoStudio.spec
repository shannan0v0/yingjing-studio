import os
# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['studio.py'],
    pathex=[],
    binaries=[],
    datas=[('u2netp.onnx', '.'), ('../assets/result.jpg', '.'), ('perch.png', '.'), ('../cat-demo.jpg', '.'), ('../forest-demo.jpg', '.'), ('notices','notices')] + ([('hd','hd')] if os.path.isdir(os.path.join(SPECPATH,'hd')) else []),
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'pytest', 'IPython', 'pkg_resources', 'setuptools'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='PhotoStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='PhotoStudio',
)

