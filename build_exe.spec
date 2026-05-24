# PyInstaller spec file — builds a single-file Windows .exe
#
# Run from INSIDE the water-contact-angle folder:
#   pip install pyinstaller tkinterdnd2
#   pyinstaller build_exe.spec

import os
block_cipher = None

# Include tkinterdnd2's native TkDND library so drag-and-drop works in the exe.
try:
    import tkinterdnd2
    _dnd_src = os.path.join(os.path.dirname(tkinterdnd2.__file__), "tkdnd")
    _dnd_data = [(_dnd_src, "tkdnd")] if os.path.isdir(_dnd_src) else []
except Exception:
    _dnd_data = []

a = Analysis(
    ['app.py'],
    pathex=['.'],          # run from inside water-contact-angle/
    binaries=[],
    datas=_dnd_data,
    hiddenimports=[
        # sibling analysis modules (lazy-imported — PyInstaller won't find them automatically)
        'analyzer',
        'calibration',
        'detection',
        'preprocessing',
        'visualization',
        'fitting',
        'evaporation',
        'advancing_receding',
        'live',
        'surface_energy',
        'baseline',
        'publication',
        'tuner',
        'segmentation',
        'reflection',
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
        'tkinterdnd2',
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
    console=False,        # change to True temporarily to see crash messages
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
