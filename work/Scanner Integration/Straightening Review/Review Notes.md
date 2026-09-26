# Receipt straightening review

The user clarified that the first preview shown in the run reported complete
is the preferred text and line result: `Baseline/Image 2.png`. The later global
refit in `Final/Image 2.png` improved the barcode but degraded text and lines,
so it must not remain the whole-page default.

`Local Combination/Image 2.png` starts from the original accepted mesh. It
transfers the later barcode correction only around detected barcode bars and
the vertical correction only around the final detected text row. All regions
come from visual evidence, with no sample-name or index conditions. Optional
refinement failure returns the original mesh.

The measured protected guide increase is effectively zero. Barcode envelope
residual is 0.040 pixels at measurement resolution; final-row residual falls
from 0.518 to 0.177 pixels. These are guide measurements, not a claim that all
visible glyphs are perfect. At full output size the body interior differs only
by at most one colour level due to floating-point resampling, with its original
geometry preserved. The user sees the same previews displayed by view_image.

The original preview remains available as the fallback. Working card rules,
commercial clearance policy and Safe Build remain protected.

## Focused final-row correction

The user's next attachment is pixel-identical to `Baseline/Image 2.png`, which
confirms the reference. The user called it much better and asked to focus on
the most bottom text. `Footer Correction/Image 2.png` solves that observed text
baseline directly at one output height, rather than accepting the later global
refit's footer compromise. Only the footer's vertical displacement changes.
Everything above output row 1430 is pixel-identical to `Local Combination`.
The measured footer residual falls to 0.00044 pixels; this measures the fitted
guide and does not guarantee perfect individual glyph contours.

The 48-image local regression passed and changes only image 2; all other 47
PNG files are byte-identical to `Preserved Baseline`. Only image 2 qualified for
local refinement. Its focused footer render passes the same protection and
fold checks. Six Python checks pass, including retaining the original mesh on
refinement exceptions. The previous full Flutter suite passed 115 tests; no
detector/card code was changed during this focused refinement.

The rebuilt New Build and installed app pass strict signature and commercial
clearance verification. Their executable SHA256 is
`351b1fbee63c2295a9ac3da44e1dc5fe0a0c5ccdbd13242d27f53b94bdb9dcc4`.
No new dependencies, model weights or contacted-license components were added.

Live app verification opened receipt 2, rendered the correction and saved its
PNG. `work/Flattened Outputs/bill_segmentation_image_project_000091 Flat
1790405663508054.png` is pixel-identical to the reviewed Footer Correction
preview, with zero channel difference.

## Actual ink and two-axis residual correction

The user rejected the fitted-guide result as imperfect, clarified that residual
detection must work anywhere, and approved the actual footer crop from
`Validated Residual/Image 2.png`. That full PNG is now the immutable reviewed
reference. It is pixel-identical to the latest `Paragraph Paper Review/Image 2.png`.
Normal letter bodies supply measured baseline anchors; ascenders, accents and
descenders cannot create unsupported deformation evidence. Exact prior mesh
evaluation and reuse of prior rendered pixels preserve all pixels outside the
accepted correction regions and the entire silhouette.

Image 5 is the next review target; image 46 remains a later challenge. The
two-axis pass groups independently supported neighbouring bends into ordered
surface patches, joins nearby fragments of the same printed row, and keeps
paragraph gaps distinct. The upper six rows and indented eight-row paragraph
qualify. Repeated right-margin evidence supplies the other-axis correction;
composition preserves the established horizontal baselines. The reviewed full
result is `Paragraph Paper Review/Image 5.png`; `Image 5 Comparison.png` shows
the prior and current upper paragraphs at the same scale. This is measured and
visually inspected improvement, not a declaration of perfect straightening.

The fresh `Two Axis Regression` checks all 48 documents at 640 pixels. Only
2, 5, 6, 25 and 39 change versus `Local Regression`; all 48 alpha silhouettes
remain exact, and the other 43 full images are pixel-identical. Confirmed rigid
IDs/cards retain the existing bypass. Images 46 and 47 remain unchanged because
their proposals fail the crossed-content or fold protections. Failed additive
patches, overlapping-row proposals and paragraph-spanning fits were discarded;
none becomes the next accepted baseline.

Eleven Python checks pass, covering position independence, optional failure
fallback, coherent paragraphs, both margins and crossed-rule rejection. The
renderer and parser changes passed all 118 Flutter tests, including exact
outside-pixel and silhouette preservation. The signed New Build and installation
pass strict signature and commercial bundle verification. Their executable
SHA256 is `e843ab7007dbd4b1a070ac5b970ec792eca0ccd293c07befe22c8e6393e67082`.
No new dependency, model or separate licensing contact was introduced.

The installed app saved `work/Flattened Outputs/bill_segmentation_image_project_000091
Flat 1790408948160099.png`, exactly matching the approved `Validated Residual`
image 2 with zero channel difference. Native image 5 verification was interrupted
by the user's active app navigation; the controls were left with the user.

An additional offline ordered single-row attempt is rejected. `Ordered Single
Review/Image 2.png` changes 17,329 pixels of the approved receipt, while image 5
receives no extra accepted correction. The candidate script was deleted; the
installed default never used it. `Paragraph Paper Review` remains the accepted
engineering checkpoint. The general straightening goal is still open: horizontal
and vertical residuals, visual perfection and the image 46 challenge are not
claimed complete.

## Protected ink checkpoint before pink invoice work
The user requested a saved Git checkpoint before further pink-document work.
The accepted engine now refines actual glyph ink, excludes false sideways-word
groups, and composes bounded corrections with the full prior map. It preserves
exact prior rendered pixels outside each patch, including nested refinements.
Glyph height, sideways displacement, neighbouring ink and topology are checked;
failed proposals return the accepted map. No sample-name or index gate is used.

The full Protected Ink Regression changes only images 5, 6, 17, 36, 39 and 47
versus Two Axis Regression. All 48 alpha silhouettes remain identical; the
other 42 images remain pixel-identical. The 1600-pixel focused image 2 is
pixel-identical to the approved Validated Residual reference. Image 5 retains
its accepted two-axis corrections and adds one supported residual row.
Fourteen Python tests and all 118 Flutter tests pass. The final test rerun
needed the tool-directory import context; no implementation change was needed.

New Build 1.0.37+49 is signed, passes strict signature and commercial bundle
verification, and matches the checkpoint source. Executable SHA256:
`3ef69576ac9ec505234201e8b367c23efff646f352d1f7bcf2d3cf1fe69a094f`.
The installed app is still the previously accepted build 48; build 49 has not
yet had native save verification. Safe Build is unchanged. No dependency,
model, license contact or unapproved asset was added.

This is a recovery checkpoint, not completion of general straightening.
The pink invoice (evaluation image 17) remains the next focus, with all
accepted corrections protected. Image 46 remains later. When finished,
replace this branch commit again, retaining exactly one commit above main.

Evidence hashes (generated images remain local evaluation artifacts):
- `Validated Residual/Image 2.png`: `fecd739c66d4e4c0877814c56586b4695e398c45e8d7879b11db9ccd36920484`
- `Protected Ink Review/Image 5.png`: `85e92fecaa10870712d123e401e1899e263ce309ee17c2531ab46b4fbd118b04`
- `Protected Ink Regression/report.json`: `b4cc81f5b864a2e7485d11bad1d380085f67714de60505a35aa83ef6364e78b2`
