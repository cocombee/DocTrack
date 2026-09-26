# Commercial Clearance

The commercial branch preserves DocTrack's application features and card-fix
source while keeping the original New Build recoverable on `main`.

## Distribution decision

All DocTrack development must preserve commercial-use and redistribution
clearance. Exclude code, dependencies, assets and model weights that need
separate contact with an owner, negotiated terms or custom permission. Use
only rights already established by documented licences; uncertain rights
require exclusion and a verified alternative, not a permission request.

The packaged commercial build may remain proprietary. Its bundled Flutter,
Dart, image-processing and ONNX Runtime components use commercially permissive
licences whose notices are present in the generated Flutter licence registry.

The release must exclude any model whose redistribution rights are unknown.
DocAligner model weights are therefore development-only until authoritative
commercial redistribution permission is recorded. UVDoc and Text Line Dewarp
are permissively licensed local research integrations but are not packaged in
the current application bundle.

Rejected FairScan, DocRes and LaMa evaluation assets and their evaluation-only
scripts are removed from the commercial branch.

## Verification

From `work/Source`, build the release and run:

```sh
dart run tool/verify_commercial_release.dart
```

The verifier fails when required notices are missing, a bundled component is
not commercially approved, an unapproved model-weight format appears, or a
rejected evaluation family is present in the application bundle.

The exact Audyt UI placement and release workflow are documented in
`Audyt Commercial Attribution Plan.md`. That plan does not modify Audyt.
