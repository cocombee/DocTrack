# Audyt Commercial Attribution Plan

This plan is maintained in DocTrack. It does not change the Audyt repository.
Audyt remains proprietary and closed-source.

## Exact placement in Audyt

Do not add a new main-sidebar destination. Audyt already opens its settings
dialog from the sidebar footer, so legal information belongs at the bottom of
that dialog.

| Audyt location | What to add | Purpose |
| --- | --- | --- |
| **Settings**, after **File Setup** | A final **About & Legal** heading with one **About Audyt** row | Keeps legal information discoverable without mixing it with editable accounting settings. |
| **About Audyt** page/dialog, header | Audyt name and icon, installed version and build number, `Copyright © [year] [legal owner]. All rights reserved.`, and `Audyt is proprietary software.` | Identifies the commercial product and owner. Replace the bracketed owner before release. |
| **About Audyt**, directly below the header | `Audyt includes commercially permitted third-party components. Their licences apply only to those components and do not make Audyt open source.` | Makes the proprietary status unambiguous. |
| **About Audyt**, **Third-Party Software** row | Opens a searchable/read-only list generated from the installed manifest. Each row shows component name, version or commit, copyright owner, SPDX licence, upstream URL, and a **View licence** action. | Gives every distributed dependency its required notice. |
| **About Audyt**, **Document Scanning Acknowledgements** row | Opens the scanning-specific subset of the same manifest. | Keeps DocTrack/scanning attribution easy to find without duplicating legal text. |
| **About Audyt**, **View all licences** row | Opens the exact `THIRD-PARTY-NOTICES.txt` included in the installed build, with copy and search controls and offline access. | Provides the complete authoritative notices. |

### Intended Audyt implementation map

These are future Audyt changes. This commercial-clearance branch must not edit
the Audyt repository.

- `Audyt/lib/app/naming_setup.dart`: append the **About & Legal** heading and
  **About Audyt** row after the existing **File Setup** settings group.
- `Audyt/lib/app/about_legal.dart`: own the About page/dialog, manifest loader,
  component list, scanning subset, and full-notices viewer. Keep this separate
  from editable accounting settings.
- `Audyt/assets/legal/THIRD-PARTY-NOTICES.txt`: store the complete licence and
  copyright notices for the exact installed release.
- `Audyt/assets/legal/third_party_manifest.json`: store the structured source
  for the two component lists.
- `Audyt/pubspec.yaml`: bundle both legal resources explicitly.
- `Audyt/tool/verify_commercial_release.dart`: fail packaging when the manifest,
  required notices, commercial approval, or distributed-file hashes do not
  match the release bundle.
- `Audyt/test/app/about_legal_test.dart`: verify the Settings entry, proprietary
  statement, version/build display, scanning subset, full notices, and offline
  readability.

Do not expose internal evaluation folders, rejected experiments, source-code
history, or research notes in About & Legal. Keep that provenance in internal
release records. The user-facing list contains only components actually shipped
in the installed Audyt release.

## Content of Document Scanning Acknowledgements

- DocAligner: Apache-2.0, only after model-weight distribution rights are
  verified in writing or from authoritative upstream licence material.
- UVDoc: MIT, including its copyright and MIT notice when its code or model is
  distributed.
- Text Line Dewarp (`PINTO0309/doc-dewarping`): MIT, including its copyright and
  MIT notice when its code is distributed.
- ONNX Runtime and Flutter/Dart packages: include the licences emitted by the
  release dependency and Flutter licence registries.
- Omit every system that is absent from the installed release. Do not list an
  excluded research model as though Audyt distributes it.

## Files bundled with Audyt

Place a generated `THIRD-PARTY-NOTICES.txt` in the application resources. It
must contain the complete notices for every bundled dependency and model. The
About & Legal page should read this installed resource rather than keeping a
second manually maintained copy of the legal text.

Keep a machine-readable `third_party_manifest.json` beside it with:

- component name and version or commit;
- upstream URL;
- licence SPDX identifier;
- copyright holder;
- distributed files and SHA-256 hashes;
- model and training-data licence status, where applicable;
- date and evidence used to approve commercial distribution.

## Release gate

Audyt must not bundle a component when its code, model weights, training data,
or redistribution terms are unknown. A permissive source-code licence does not
automatically clear separately downloaded weights.

Before every commercial release:

1. Generate the dependency and model manifest from the exact release bundle.
2. Fail the build for GPL/AGPL, non-commercial, research-only, unknown, or
   missing licences unless an approved written exception is attached.
3. Verify every required copyright, licence, and NOTICE entry is present.
4. Compare file hashes against the approved manifest.
5. Open About & Legal in the packaged application and confirm all notices are
   readable offline.
6. Archive the manifest and licence evidence with that release.

## Current DocTrack status

- Rejected FairScan, DocRes, and LaMa evaluation assets are excluded.
- UVDoc and Text Line Dewarp have permissive MIT source licences.
- DocAligner source uses Apache-2.0, but its separately downloaded model-weight
  distribution scope remains unresolved. Those weights are not commercially
  approved until that evidence is resolved or the models are replaced.
