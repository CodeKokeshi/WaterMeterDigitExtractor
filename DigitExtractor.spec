# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for DigitExtractor
Builds standalone executables for Windows (.exe) and macOS (.app)
"""

import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect PyQt6 data files
pyqt6_datas = collect_data_files('PyQt6')

# Hidden imports needed for OpenCV and PyQt6
hidden_imports = [
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'cv2',
    'numpy',
]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=pyqt6_datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        '_tkinter',
        'PIL',
        'torch',
        'tensorflow',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DigitExtractor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window (GUI app)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='icon.ico',  # Uncomment and add icon file if available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='DigitExtractor',
)

# macOS .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='DigitExtractor.app',
        icon=None,  # Set to 'icon.icns' if you have a Mac icon
        bundle_identifier='com.digitextractor.app',
        version='1.0.0',
        info_plist={
            'NSPrincipalClass': 'NSApplication',
            'NSHighResolutionCapable': 'True',
            'CFBundleDocumentTypes': [
                {
                    'CFBundleTypeName': 'Image',
                    'CFBundleTypeRole': 'Viewer',
                    'LSItemContentTypes': [
                        'public.image',
                        'public.jpeg',
                        'public.png'
                    ]
                }
            ]
        },
    )
