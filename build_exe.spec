# PyInstaller spec file — builds a single-file Windows .exe
# Run with:  pyinstaller build_exe.spec
#
# Prerequisites:
#   pip install pyinstaller
#   (run from the PARENT directory of this package)

import sys
from pathlib import Path

block_cipher = None
pkg_dir = Path('water-contact-angle')   # adjust if your folder name differs

a = Analysis(
    [str(pkg_dir / 'app.py')],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'cv2',
        'numpy',
        'matplotlib',
        'matplotlib.backends.backend_tkagg',
        'PIL',
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
    console=False,       # False = no console window (GUI app)
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
