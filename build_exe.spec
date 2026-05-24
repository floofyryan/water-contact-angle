# PyInstaller spec file — builds a single-file Windows .exe
#
# Run from INSIDE the water-contact-angle folder:
#   pip install pyinstaller
#   pyinstaller build_exe.spec
#
# To see error messages (debug build):
#   change  console=False  to  console=True  below, then rebuild.

import os
block_cipher = None

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        # sibling analysis modules — listed so PyInstaller compiles them in
        'wca.analyzer',
        'wca.calibration',
        'wca.detection',
        'wca.preprocessing',
        'wca.visualization',
        'wca.fitting',
        'wca.evaporation',
        'wca.advancing_receding',
        'wca.live',
        'wca.surface_energy',
        'wca.baseline',
        'wca.publication',
        'wca.tuner',
        'wca.segmentation',
        'wca.reflection',
        # third-party
        'cv2',
        'numpy',
        'scipy',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'PIL',
        'PIL._imagingtk',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ContactAngleAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,      # ← set to False for final release (hides terminal window)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
