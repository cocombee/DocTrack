# Document Shape Reconstruction

This is the target behaviour and acceptance contract. It is not a claim that the current detector satisfies every case.

## Decision pipeline

Select the shape family before local fitting: supported ID-card, name-card or passport evidence selects a four-main-side model with independent sharp/rounded/cut corners. Type evidence does not force all corners to be rounded. Unknown type grants no extra deformation freedom.

Start with four shared anchors (two endpoints per side). A sustained smooth bow uses Bézier handles without additional boundary anchors. Add an anchor only for a separately supported change of direction. A short inset/outset that departs and returns without supported joins is a detour; a continuous curve or supported fold is not. Frame closure is a separate constraint, never inferred from black pixels.

1. Start with the simplest four-sided hypothesis, then classify the document as rectangular, rounded, or custom/irregular before local refinement. Image 35 is a custom-shape regression example (the former sample 47 was excluded at the user’s request); they must not be forced into a rectangle. Perspective changes angles and apparent corner radii. A category is a hypothesis, never permission to overwrite visible geometry.
2. Assess each corner independently: hard, rounded, or unresolved. Mixed corner types are valid. Hard corners use one anchor; a rounded transition normally uses two anchors joined by a tangent-continuous Bézier arc.
   Without physical evidence of deformation, a side continues to its end corner: a sharp L junction or a rounded transition between the two adjoining sides. An inward excursion followed by a return to the same side is not a corner and is forbidden as incidental refinement. Actual folds, crumpling, tears and custom cuts require separate perimeter evidence; stronger contrast from internal printing never grants that exception.
   If the fitted sides miss the visible corner angle, an additional anchor may make a small local inward correction. Bound it to the corner's localization uncertainty; do not turn a failed intersection into a deep inset. A larger cut or deformation requires independent physical evidence.
3. Assess wrinkles, curvature and folds (berkedut/lipatan). Preserve sustained bows, real crease direction changes, and visible fold outdents. Ignore tiny tears, ragged microtexture and isolated speckles. An interior crease does not necessarily change the perimeter.
4. Jointly assess occlusion and irregularity. A hand, pen or staple can hide a regular edge; it must not create a dent in the inferred paper. Continue supported adjacent edges behind the object. Do not classify every unusual shape as occlusion. Where evidence cannot distinguish the two, report uncertainty.
5. Assess camera clipping separately. If paper continues beyond the photo, the image boundary closes the visible selection. It is not a physical paper edge or a paper corner. Never invent geometry outside the available image.
6. Validate the physical perimeter. Reject internal printing, rules, QR codes, coloured panels, shadows and unrelated background edges. Transparent laminate and the paper inside it are distinct possible boundaries; ambiguous selection must be flagged.
   Colour changes inside the document are not boundaries. Coloured fills, panels and printed graphics remain part of the same paper surface when continuity and the document projection support them. Background or covering objects may be disconnected, overlap the paper, or lack a supporting paper surface; connectivity alone is not decisive because touching objects and full-bleed printing exist. Do not require paper to be white.
7. Fit the simplest geometry supported by the validated perimeter. Straight lines for straight edges, real curves for sustained curvature, independent tangents at folds. Reject crossings, reversed tiny arcs and invalid coordinates. Few points are a consequence of a correct fit, not a reason to erase folds.
8. Apply the four-source-pixel inward selection margin only after physical geometry is established. Camera-frame sections receive zero inset in the frame-normal direction. A join with a real edge remains on the frame while following that real edge's inset. Preserve the physical boundary separately for stage-two remapping.

## Evidence and interfaces

- Run from original upright image pixels, without saved annotations, filenames, dataset numbers or previous predictions as detection inputs.
- Separate image evidence, shape hypotheses, occlusion inference and display inset. Record uncertainty rather than treating a hypothesis as observed truth.
- Object recognition needs validated identification evidence. Colour heuristics alone are not a built-in hand/object recognition model; do not describe them as one.
- Return normalized physical geometry, explicit corner/edge classifications and confidence/reason codes. Keep the core portable, with optional model adapters for the mobile-first Akaunting transaction upload editor.
- Stage two flattens/remaps the document and normalizes text; it is separate from this boundary detector.

## Review and regression

- Preserve all original images, manual anchors and handles. Back up and hash reference data before tests.
- Generate fresh predictions for a requested full test; never present cached results as a fresh run. Normal navigation may retain its cache and global engine-line visibility.
- Inspect all edges and enlarged corners, including frame intersections. Contact sheets and overlap scores alone cannot establish precision.
- Compare approved images 5, 6, 7, 43 and 44 for regressions, while retaining the user's later corrections. Add mixed corners, internal borders, coloured panels, QR noise, folds, occlusion and camera clipping to synthetic checks.
- Keep candidate and installed results identifiable. Do not promote an unverified candidate or call a visually rejected result a success.
- Release comparisons must sample the entire perimeter at uniform source-pixel arc length, in both directions. Checking only line endpoints can miss a severely misplaced side. Use `work/Source/tool/boundary_regression.py`; a zero exit status is only absence of measured regression, not visual approval. Any blocked image prevents promotion; mean improvement cannot compensate for a broken sample.
- Retain versioned source and prediction checkpoints before experiments. A rejected experiment must not become the next presumed-good baseline.
- Output the selected line/curve primitives directly. A generic curve fitter must not rebuild them afterward. Image evidence may validate the extent of a proposed document, but must not be copied into an unrestricted output contour.
- OCR content is supporting evidence, not a validated semantic card identifier. Repeated transaction fields and amount rows can support paper classification; punctuation alone cannot. Multiple identity fields can support the rigid-card hypothesis. Unknown content remains unknown.

## UI constraints

Keep stroke, anchor and handle sizes constant while zooming. Preserve box selection, curve-handle reveal, matching toolbar button sizes, regular-weight paper names, logo spacing and native window controls. Detection remains the third toolbar action and executes the engine.

## Approved reference and shape guardrails — current repair

- The saved green references are the evaluation target. Preserve their geometry and handles. The newly requested crop rule is an addition; it does not justify flattening their bends or redesigning their corners.
- A verified rigid quadrilateral has four main straight sides and at most four corner arcs. One or two independently supported cuts may add short straight sides. Weak contrast is uncertainty, not permission to generate an unrestricted contour. This is a geometric hypothesis; the current adapter has no validated semantic ID-card classifier.
- Crop limits must come from the actual raster extent or an explicit valid-data polygon supplied by a trusted image transform. Black pixels, even pure and uninterrupted, are image data unless source metadata establishes otherwise. Black printing must never establish a crop limit. Cropping only affects the supported frame closure, never unrelated document edges. Verified clip segments receive zero display inset.
- A separate FairScan v1.2.0 segmentation evaluation was rejected because its raw masks merged background objects, followed hands, and omitted printed card regions. Its model, outputs, and evaluation scripts were removed; it is not part of the production adapter.

## Extension preserves corners

After corner type and radius are accepted, extending a card side changes the side position and length. Carry corner offsets and curve handles to the extended corners in the adjacent-side coordinate basis; do not redetect corners or scale their radii merely because the width changed. The Affinity rectangle demonstration is the interaction reference. Two supported rounded corners can constrain a card/passport to four main sides without requiring all four corners to be rounded.

Use confidently supported edges as the alignment guide. Preserve their position and corner geometry when extending an uncertain side. Near-frontal corner sizes should be familiar, not forced pixel-identical: allow a modest range for perspective and localization. Extension translates a side in parallel instead of independently moving its endpoints. Strong perspective requires its existing projected directions to remain intact.

If the first candidate edge is uncertain, cross-check the second and third independently supported edges before extending or relocating it. Weak local contrast must not override their shared shape and perspective. If those edges disagree, retain uncertainty rather than inventing a detour.

## Printed guides and conservative straight-side recovery

For paper, infer uncertain corners from confident incident edges and test their direct connection before adding deformation. A narrow fallback may retain the direct side when the other three sides verify and both foreground surface continuity and boundary evidence support it. Keep accepted ID/card rules unchanged.

Internal printed rules may corroborate paper folding after accounting for perspective. Require multiple distinct rules (currently at least three) to bend coherently near the same position before treating them as fold evidence. A single graphic is insufficient; absent or ambiguous printing must abstain. Printed rules never become crop boundaries. The current implementation uses coherent printed folding to veto the straight-side fallback; it does not infer new perimeter anchors from printing.

Validate candidates against every active sample before building. Build37 changes only sample17 against accepted35; all other48 outlines match exactly. Sample numbers and reference labels are evaluation inputs only, never runtime detection rules.

## Final ID shape completion and clipping (Affinity rule)

Resolve the document family and physical perspective before emitting flexible side primitives. Strong opposite rounded corners can require revisiting an earlier unknown/paper classification. A weaker existing rounded corner may use a stronger adjacent one as a bounded hypothesis; preserve genuinely sharp corners. Side intersections determine uncertain corner locations, so rounding does not mask a top-left inset or bottom-right outset.

Complete the ID's predicted geometry before intersecting it with the raster or explicitly supplied original crop polygon. Let the completed corner fall outside. Split curves at their exact crop intersections, preserve the visible Bézier geometry, and join on the crop boundary. Do not squeeze an entire rounded corner inside or shift its handles to the frame. Fully visible geometry is unchanged by this operation, even if a control handle lies outside while the curve remains inside. Crop closure gets no normal inward inset. Pixel colour alone does not establish an original crop.

Current scope excludes receipt37 at the user's request. Preserve the legacy receipt/paper branch while the ID/explicit-crop path uses final clipping. Build38 regression changes only29 and45;41 and all other47 shapes remain geometrically unchanged.


## Reconstruction purpose and final geometry constraints

The user clarified that the engine ultimately reconstructs a flat document from the photographed card: undo perspective, bends and surface curvature while preserving original proportions and corner shape. Outline detection supplies geometry for this reconstruction, not merely a visual crop. Preserve real deformation evidence for downstream dewarping; a perimeter alone does not establish the interior warp. This clarification is design intent, not a request to implement the flattening stage now.

Rounded corner sizes must be compared in the document plane, accounting for projected local scale. Use stronger corner evidence to bound weak corners; retain genuinely sharp corners. When the final shape changes from paper to rigid card, derive its corner tangents from the final card sides, not discarded paper detours. Final inset corner handles must not reverse past the adjoining edge direction or overshoot their tangent intersection. Genuine curved paper sides remain eligible.

For the hand-covered photo ID, top portrait padding is a bounded supporting estimate for right padding. Require physical card-to-dark-gap-to-hand evidence before moving the side, cap movement by measured top padding, and preserve perspective. The dark gap is outside the card but is not an image crop boundary. Use source filenames for sample identity after exclusions.

Two or three clearly visible corners establish a conservative corner-family prior for obscured corners, in either direction: sharp predicts sharp, rounded predicts rounded with perspective-adjusted radii. Covered or weak corners are unknown, not independent votes. Preserve strongly visible exceptions and mixed-corner documents. Current implementation rejects weak arcs with at least two visible sharp corners and completes low-contrast hidden corners from at least two strongly supported rounded corners.

Workflow correction: after photo-margin side recovery, do not rerun corner classification. Carry accepted arcs and handles in the adjacent-side coordinate basis, rebuild the corrected four main sides, and replace the former occlusion detour with their sharp intersection. Final clipping and inset follow this step. This prevents a covered corner from gaining a new arc after its side is fixed.

Paper cut classification must verify contrast along the entire proposed short diagonal, not only its average. A strong crossing from background structures can otherwise create a false bevel at a sharp corner. Apply this check in the verified paper-side primitive path; preserve the rigid-card path even when OCR is unknown. Evaluate changes against all accepted samples.

## Active flattening phase and orientation

Flattening is now explicitly requested. First isolate the detected document, then undo perspective, skew and surface deformation using reliable text, printed lines and shapes. Estimate the format (paper families, passport, identity/credit/business card) when evidence supports it; otherwise retain estimated proportions. Do not invent text. Repair genuine missing paper edges with nearby stock colour, with a feather that protects printed detail. Preserve transparent exterior pixels for originally rounded documents: two equal rounded corners plus two sharp corners, or four equal rounded corners in the flat plane. Custom corner geometry is for IDs only.

The fourth toolbar button is a global flatten-mode toggle across sample navigation; a second click disables it. It uses the existing canvas, without extra panels or controls. Auto-orient the final image using reliable word direction and portrait evidence. Ambiguous or conflicting evidence retains the existing orientation. OCR recognition confidence alone cannot establish direction because Vision can read rotated text internally; use recognised-letter positions. Exact quarter-turn rotation must preserve pixel values and alpha.

A little bottom content may move beyond the output frame when necessary to straighten sample 46; do not force every pixel back inside by bending the mesh. This is permission for bounded cropping, not permission to erase text or use sample-specific runtime logic. All new geometry remains subject to fold-over checks and preservation of existing working cases.

Accepted two/four rounded-corner families survive flattening even when OCR or estimated physical proportions do not identify a card format. Format inference must not turn those accepted curves into opaque rectangular corners. Confirmed transaction paper still repairs missing edges to a rectangle.

## Receipt content straightening

A ratio-only card estimate cannot suppress interior correction without at least
two true rounded corners or two distinct short two-point corner transitions.
Long bowed paper sides are not corner evidence. Confirmed card and passport
evidence retains its rigid perspective path, even with conflicting OCR.

Straightening must assess printed rules, text baselines and barcode bars and
envelopes. Dark-only text extraction may recover rows lost when opposite ink
polarities merge; it supplements validated guides rather than replacing them.
Barcode envelopes must retain observed curvature before solving. Apply the
narrow receipt refinement only with multiple independent guides, a valid
nonfolding mesh and per-guide preservation checks. Failed proposals retain the
previous accepted mesh. Sample names and numbers are evaluation inputs only.

User visual review supersedes aggregate guide scores. The first preview in the
September 26 straightening run is the receipt's accepted text/line reference;
the later global refit was rejected because its improved barcode came with
worse text and lines. Preserve the original page map and transfer the improved
map only inside evidence-derived barcode and final-row regions. Transfer only
vertical displacement for the final text row. Verify all protected original
guides and the full regression set, and retain the original map on failure.
Never claim perfect straightening from guide scores alone.

Straightening is a general residual-detection process, not a request to repair
a named top, middle, centre or bottom region. Start from the accepted map and
inspect reliable text, rules, barcode geometry and shape landmarks throughout
the document. Actual letter ink must support a text correction; a straight
polynomial fitted to an incorrect baseline does not prove that the letters are
straight. Reject font-contour scatter that lacks a coherent distortion trend.

Propose bounded corrections only at evidence-derived locations. Validate every
other accepted guide, reject folds, and preserve the document silhouette. Each
failed proposal leaves the prior result in place. Evaluate the exact prior map
outside accepted regions and reuse its rendered pixels there, including border
repair, so a local correction cannot change established content elsewhere.
Verify position independence with the same distortion at multiple page heights
and run the full document regression. Visual review remains authoritative.

Evaluate both document axes. Repeated text margins can support a horizontal
displacement correction for vertical alignment; compose it with the accepted
surface map so established horizontal baselines and spacing remain fixed.
Protect crossed printed rules, barcode bars and envelopes, and shape landmarks.
Ragged margins or unsupported alignment must not force a correction.

Neighbouring text rows with independently measured, correlated bends may share
one ordered surface patch. Join nearby fragments of the same ink row before
fitting, while keeping adjacent rows and paragraph gaps distinct. Validate the
whole patch against established content outside it. Retain the exact approved
image 2 render as a regression reference while reviewing image 5; approach
image 46 only after earlier working cases remain intact.

Actual-ink residual patches must preserve the complete prior refinement chain.
Protect neighbouring glyph envelopes and real sideways words, reject accidental
vertical stacks of paragraph glyphs, and validate glyph height and sideways
displacement as well as baseline straightness. The pink invoice is the next
review target; accepted receipt, paper and card results remain regression gates.
