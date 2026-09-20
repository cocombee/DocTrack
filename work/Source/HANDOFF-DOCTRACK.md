# HANDOFF — pen_tool → DocTrack (to Akaunting owner)

Date: 2026-09-17 19:40 (+08) · From: Admin/default chat (Hermes) · To: Akaunting project owner
Project: `Hermes/Apps/Akaunting/work/scan-corner-data/pen_tool_app` (Flutter, single file `lib/main.dart`, 701→~770 lines)

## What changed in this handoff batch (verified)

1. **Hold-to-reveal curve handles** (`_revealHandles`, `_initTangentHandles` in `lib/main.dart`)
   - Long-press/hold a dot → its two bezier handles appear in **blue** with a blue ring (non-destructive; point stays corner until a handle is dragged).
   - Dragging a revealed handle commits `smooth = true` (`_handle.onPanStart`).
   - Unpin: click empty canvas (does NOT add a dot while pinned) or `Esc`.
2. **Dot deletion made discoverable** (`_deleteRevealed`, hint text)
   - Right-click a dot = delete (pre-existing, kept).
   - NEW: hold a dot → handles reveal → press `Delete`/`Backspace` to delete it (guard: ≥4 dots, `Cmd+Z` undo works).
   - Sidebar hint line added under status.
3. **Rename pen_tool → DocTrack**
   - `PRODUCT_NAME = DocTrack` (all 3 configs) + TEST_HOST fixed in `macos/Runner.xcodeproj/project.pbxproj`.
   - Sidebar title `Pen tool` → `DocTrack` in `lib/main.dart`.
   - Flutter still emits `pen_tool.app` from pubspec `name:` — bundle is post-processed: exe renamed to `DocTrack`, `CFBundleExecutable/Name/DisplayName = DocTrack`, re-signed ad-hoc.
   - Installed at **`/Applications/DocTrack.app`** (running PID 56242 at handoff; window title verified "DocTrack").
   - Old installs removed: `/Applications/pen_tool.app`, `/tmp/pen_tool_build` instance.

## Evidence paths
- Source: `pen_tool_app/lib/main.dart`, `pen_tool_app/macos/Runner.xcodeproj/project.pbxproj`
- Build: `pen_tool_app/build/macos/Build/Products/Release/DocTrack.app` (39.7 MB, Release)
- `flutter build macos --release` → ✓ built; `flutter analyze` → 0 errors (15 pre-existing warnings: public `notifyListeners` misuse + unused `_loadedIndex` — pre-existing, untouched)

## Known follow-ups for Akaunting owner
- **TCC**: first launch from the new bundle path may prompt for access to the Google Drive folder — if images/labels don't load, grant Files access (System Settings → Privacy → Files & Folders /Folders) and relaunch.
- **Bundle ID** is still `com.example.penTool` — harmless, but rename in `macos/Runner/Info.plist` + project if you want a clean identity (requires fresh codesign).
- pubspec `name: pen_tool` was deliberately NOT renamed (it drives the build product name; renaming breaks the Flutter build step). If you want the emitted bundle to be DocTrack natively, change pubspec name and re-run the whole pipeline, then delete the post-process step.
- `/Applications/pen_tool.app.old-1933` was deleted earlier this session; no other backups exist.

## How to rebuild (for the owner)
```
cd "Hermes/Apps/Akaunting/work/scan-corner-data/pen_tool_app"
flutter build macos --release
# then: ditto build/macos/Build/Products/Release/pen_tool.app → DocTrack.app,
# rename exe + CFBundle keys, codesign --force --deep -s -, install to /Applications
```
