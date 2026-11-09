# Xcode installation verification — 2026-11-06

- Host: m5pro, macOS 27.2 (26B123).
- Source: /Users/qklj/Downloads/Xcode_27.2.xip. Retained.
- Package signature: pkgutil reports signed Apple Software.
- SHA-1: 99f19f73742c872cde03d02628a058688a37f7a1, matches downloaded release index.
- Installed app: /Applications/Xcode.app, Xcode 27.2 (27B52).
- License accepted; first-launch components installed successfully.
- Active developer directory: /Applications/Xcode.app/Contents/Developer.
- macOS SDK: 27.2.
- Metal Toolchain: 27B47, installed through xcodebuild -downloadComponent metalToolchain.
- Metal compiler: 32023.922 (metalfe-32023.922.4).
- First Metal catalog request failed; second request downloaded and installed successfully. Both logs retained.
- Actual compile/link passed: smoke.metal -> smoke.air -> smoke.metallib, Metal language 4.0, deployment target 26.2.
- xctrace lists Metal System Trace and other profiling templates.
- verification.log contains final command outputs.
- ImageWAM production Python environment, models and inference code were not modified.
- This verifies the toolchain installation only, not the previously blocked full MLX source builds or inference benchmarks.

Reproduce:
```sh
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cd /Users/qklj/work/cx/ImageWAM_MLX/records/xcode_install_20261106
xcrun -sdk macosx metal -std=metal4.0 -mmacosx-version-min=26.2 -c smoke.metal -o smoke.air
xcrun -sdk macosx metallib smoke.air -o smoke.metallib
xcrun xctrace list templates
```
