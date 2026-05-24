# PyInstaller spec file — builds a single-file Windows .exe
#
# Run from INSIDE the water-contact-angle folder:
#   pip install pyinstaller windnd
#   pyinstaller build_exe.spec

import os, glob
block_cipher = None

# Collect all sibling .py analysis modules so they are bundled as plain files
# that can be found via absolute import (import analyzer, import detection, …)
_here = os.path.dirname(os.path.abspath('app.py'))
_sibling_modules = [
    'analyzer', 'calibration', 'detection', 'preprocessing', 'visualization',
    'fitting', 'evaporation', 'advancing_receding', 'live', 'surface_energy',
    'baseline', 'publication', 'tuner', 'segmentation', 'reflection',
]
_extra_datas = [(os.path.join(_here, f"{m}.py"), ".") for m in _sibling_modules
                if os.path.exists(os.path.join(_here, f"{m}.py"))]

a = Analysis(
    ['app.py'],
    pathex=['.'],
    binaries=[],
    datas=_extra_datas,
    hiddenimports=[
        # sibling analysis modules
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
        'windnd',
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
    console=False,        # change to True temporarily to see crash messages
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
