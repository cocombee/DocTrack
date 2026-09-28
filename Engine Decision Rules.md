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

Trace printed rules as locally supported visible sections. Text, weak stock,
crossings and missing ink must interrupt measurement; never force a continuous
ridge path through letters. Separate sections may still belong to one layout
row. Where that association is supported, give the whole row one shared target
in the document plane, using text baselines and validated rule sections as
separate measurements. Already straight sections may need rigid translation
to reach that target. Keep glyph height, spacing, crossing-rule normal position
and shape intact; do not manufacture a line or replace source pixels.

Validate the complete row, including every visible section and its joins, before
accepting a correction. A flat surviving subset does not establish a flat row.
Reject kinks, unaligned section endpoints and traces that jump onto text even
when an aggregate score improves. A failed shared-row proposal retains the
whole previously accepted result.

Develop the next correction as a separate step on the accepted result. Keep
the full working map and rendered pixels as the recovery base; do not refit
earlier successful stages to make a later stage easier. Promote only the new
bounded patch after its own evidence and all preserved content pass together.
Rejected additions must be removed from the active path before the next trial.

Individually straight sections can still occupy different levels of one row.
Associate them only through complete native word bodies, compatible font and
baseline evidence, and an unambiguous complete registered anchor. Retarget only
the fully matched misaligned section. Keep immutable source points and every
unrelated rule, glyph, join and column protected. Do not invent ridge points in
the text gap or refit the working anchor from raster alias noise.

Blank-stock capacity must cover the actual source span between neighbouring
query columns. A diagonal cell can cross an already compressed strip that both
column endpoints avoid. Broadening a global radius or increasing an area reserve
without tracing that footprint is insufficient. Retain the independent continuous
composition proof after allocation; its existing area threshold stays unchanged.

Complete material traces need both uniform mesh knots bracketing their endpoints.
Fit registered points at their actual interpolation weights when retained inverse
coordinates fall between raster knots. Sampling onto knots and back can retain a
small bend. Incompatible interpolation constraints must decline the proposal;
they do not justify loosening the full-rule straightness limit.

When a shared correction retargets one earlier complete source stroke, native
raster ridge positions remain estimates of that stroke. A small join may use
its immutable material registrations only when both complete native sections
match one unique earlier stroke, with near-exact tangent origins and no word
occlusion. Missing or ambiguous evidence keeps the legacy check. Report both
raster and physical gaps, apply the existing join limits to the physical source
points, and require a fresh native render before visual acceptance.

Keep independent native word owners separate when nearby glyph components are
connected to frame ink. Measure supported perpendicular junctions before treating
that frame ink as part of a word; full independent word envelopes retain priority.
When transposing geometry for a vertical pass, retain the original text reading
direction for unobserved glyph associations. Stacked text rows must not become
one sideways word owner merely because the correction axis was transposed.
Use the actual bilinear cells for inverse geometry checks, with their analytic
Jacobian and residual-decreasing steps, rather than a smoothed node gradient.
Blank-stock relaxation may repair a new patch only around its failing cells,
inside its actual displacement, with all measured ink constraints retained.
Keep established rule raster corridors outside the newly published region and
validate their pixels as well as geometry. None of these additions may relax the
existing topology, whole-word, barcode, shape or established-normal thresholds.

When a measured line has a bent section beside an established straight section,
anchor the correction to their measured join and exclude the straight section
from the patch. Do not use the entire line's mean or a distant crossing as the
new target. Validate the complete physical line and its joins after rendering;
a straight isolated segment is insufficient. A user-reported broken join rejects
the whole proposal even when the local geometry and word checks passed.

Verify inverse results by mapping them back through the actual bilinear mesh.
If local Newton steps stall, solve the containing source-space cells and require
a unique root with a verified forward residual. Missing or ambiguous roots fail
closed. This changes the measurement method, never its preservation thresholds.

A shared native text-row tilt may be corrected by one proper rigid rotation
when independently measured adjacent word quads agree. Preserve glyph lengths,
pairwise distances and word spacing; scaling, shear and arbitrary deformation
are forbidden. Verify proper rigid fits and physical pair distances independently
of the proposal. Keep every unselected word and established rule fixed. Record
the older translation-only metric as well: rotation changes axis-aligned height,
so that metric alone cannot establish physical glyph deformation.

An established straight rule may anchor its incomplete continuation only where
the measured traces overlap and agree. Exclude the working anchor from the new
guide, decline distant or curved anchors, and verify the whole visible join.
Limit the patch to the supported continuation and its adjacent stock; unchanged
lower pixels must be proved in the actual render before accepting the preview.

Convert native word geometry from image pixel edges to the exact measurement
grid's sample-index coordinates before combining it with independent ink. Use
native word envelopes to stop line tracking where component heuristics mistake
connected letters for a printed rule. Preserve barcode bars, their complete
envelopes and circular marks during a word-only step.

When a compact word has independently supported glyph tilt, a proper rigid
rotation may use a bounded smooth flow through adjacent blank stock to avoid
folding against fixed neighbouring words or rules. This does not permit glyph
deformation or relaxed topology: all independent words, pairwise distances,
complete rules, joins, shapes and actual rendered pixels still need their usual
checks. A failed flow retains the accepted map. A completed native job that
omits or regresses previous accepted corrections is not a quality pass.

Retain source ink for every newly established text row or glyph-supported word.
Later stages must keep that complete corrected group fixed. Preserving each
word's shape separately is insufficient: different rigid translations could
undo an already accepted shared baseline or word spacing. Apply this guard in
both correction axes and validate it independently after every new stage.

Run release regression comparisons with the app's actual mesh-selection policy
and the same measurement/render settings as the accepted baseline. Diagnostic
meshes intentionally applied to rigid documents are separate experiments. Keep
their failed comparison evidence and rerun the comparable app policy; never
change working output merely to match an incorrectly configured harness.

For document corrections, put the transition deformation in supported blank
paper. Flat stock may stretch to absorb a correction; text, barcodes, stamps,
outlines and junctions may not absorb accidental deformation. A straight rule
may move tangentially along its own direction while its normal coordinate stays
fixed. Taper or stop this freedom before a word, crossing or other shape whose
geometry would change. This permission is limited to documents and does not
authorize stretching IDs or other rigid objects.

Protect neighbouring measured rules during each local stage, including rules
not yet selected for straightening. Keep these guards temporary: preserving an
existing bend during one step must not freeze that bend as a target forever.
Compare a reported regression across the saved maps to identify the responsible
transition before adding another correction. Check the complete affected line
and its connections, not only the selected fragment.


Document coordinate composition (optional research, September27):
Keep ordered coordinate corrections as separate maps through the accepted
parent. A collapsed sampled approximation is not topology evidence. Blank
stock capacity may distribute compression between measured fixed ink bands;
equal endpoint displacement means rigid translation and may not be refitted.
Validate continuous composed area against every actually owned retained layer
with the existing >0.2 gate, as well as full word shapes, rule normals, joins,
alpha and unchanged pixels. Legacy source-grid refiners must reject retained
coordinate layers they cannot interpret. Exact pixel crops may add missed
native word evidence, but must retain prior ownership and reject clipped words.
No image-index exceptions, fixture-based production geometry, new model or
license contacts. These options remain outside the product default until raw
reproduction and all requested regression gates pass.

Optional coordinate research now keeps measured rule corners on mesh knots:
tangent denominators are multiples of the measurement width minus one, and
normal denominators are multiples of twice the measurement height minus one.
This prevents subpixel averaging from adding a new wave to an otherwise level
rule. The optional rule error gate is .03 measurement pixels; established
source rulers, text groups, complete word shapes and physical joins retain
their independent checks. Legacy/default resolution and .6 rule gate remain
unchanged until the full raw pipeline and app regression set are accepted.

Adjacent measured fragments may form one correction only when real gap ink,
safe stock, coherent endpoints and word exclusion support their connection.
A second detection pass may reveal a perpendicular continuation after its
crossing's missing ink has been measured. It does not invent missing ink or
remove word/shape protection. Inspect actual rendered stroke residuals too:
a zero map error is not sufficient to claim visual perfection.

Actual retained-map stock capacity must come from the separate owned maps,
not their collapsed grid or stored proof metadata. Keep uniform spacing in
stock whenever its actual capacity can safely absorb the correction; cap only
the intervals that need redistribution. A .21 allocation margin does not
replace the independent final continuous-area gate>.2. Rigid ink bands keep
their exact displacement. A whole owner intersecting a correction needs its
complete body inside the influence region; clipping its body cannot authorize
a partial-word deformation. Preserve the remaining shape and area gates.

Recognition can omit earlier faint text. Keep its paired source/reference
cloud as an active owner as well as an absolute shape check. Newly observed
independent words also need durable pairs for later passes; ambiguous overlap
keeps its earlier conservative constraint rather than silently merging owners.
A current OCR-only pass cannot override a failed earlier-word check. All these
changes remain optional research pending raw reproduction and app verification.

Optional verified-crossing tracking may mask actual retained word-ink pixels
with a localization halo instead of filling the whole envelope with ink.
Empty stock inside an envelope is not physical glyph evidence. This choice
changes only the measurement mask: keep the complete word's source/reference
geometry and rigid-owner constraints during correction and independent audit.
Existing envelope/default behaviour and real glyph-crossing rejection remain.

Preservation includes relationships between independent words. Rigid glyph
shapes alone can still form a newly bent text column after unequal translations.
Where≥3 distinct native word rows have a consistently vertical leading edge
and compatible font scale, retain their relative normal coordinates. Associate
only unique current owners; ambiguity cannot authorize merging or losing
earlier constraints. Keep accepted column source anchors and original references
through subsequent recognition and map transposition. A common whole-column
translation is allowed only with all other independent shape/ruler/row/area
checks; unequal translations that bend an established column are rejected.
A stock region that reaches another owner's halo must include that complete
owner before moving it. Expanding supported blank stock never waives its
existing ink, line, join, alpha or actual composed-area checks.

Optional exact-coordinate refinement can recover a complete thin rule from two
separated, independently measured fragments on the same connected component.
Separate only measured perpendicular ink outside complete word envelopes;
require thin, unbranched column profiles, real connected ink and agreement with
both fragments. Duplicate overlapping detections, missing ink, glyph exclusion
or thick/branched components cannot authorize a continuation. Every observed
trace point and physical join remains subject to the existing final checks.

This path may use one measurement pixel of tangent activation padding rather
than a fixed fraction of page width. The complete ink body, source word shape,
normal margins and all continuous-area/line/row/column/join checks stay intact;
blank stock in an extra four-pixel tangent halo is not itself glyph geometry.
The optional measured padding requires coordinate composition and at least one
pixel. Legacy defaults remain unchanged. No new model or license contact.

Repeated short horizontal divider fragments inside a native word observation
may be assigned to the independently measured printed rule only when separated
from multiple tall glyph bodies by real stock and supported by repeated trace
agreement. Keep every original point/index in either glyph or rule role, with
paired source evidence. Single components, punctuation, glyph contact, distant
rules and ambiguous evidence remain word-owned. Never rewrite an earlier
accepted word registry to make a new candidate pass.

Before registering fresh native ink as permanent word geometry, distinguish a
complete connected glyph from a small edge borrowed by a neighboring envelope's
halo. Require a dominant complete measured component and only small competing
fragments; ambiguous whole competing observations retain their conservative
owners. A tiny shared mark needs a unique undilated native envelope. Preserve
every physical point, its assigned owner, exact coordinate and source evidence.
Do not reassign ink inside an already accepted immutable word registry.

A bounded optional localization halo can recover faint observed ink just beyond
a native quad. It never changes contrast/safe-stock evidence or the quad itself.
Larger borrowed edges require stronger independent envelope evidence: a complete
measured owner, its component center inside that owner's undilated quad, and
every competing pixel strictly outside its competing quad. A pixel inside a
competing native quad keeps the ambiguity constraint. Whole source references,
all physical points and all final preservation/area gates remain mandatory.

For optional faint-edge recovery, retain baseline native localization and quads;
use a wider neighborhood only to find compact, uniquely nearby measured glyph
components not already captured by a word. Require strong ink support, preserve
all existing points, and capture the complete measured component. Do not widen
word ownership over long/thin nearby rule traces or ambiguous neighboring ink.
Record recovered component coordinates and paired source points before registry
registration. This supplements capture rather than loosening preservation.

A dense thin ruling band inside a fresh native envelope is distinct from short
repeated dashes. It may contain unequal components. Require multiple full-height
letters, two pixels of normal blank stock, thin unbranched high-density measured
components, substantial continuous coverage and independent observed rule flanks
on both sides. Keep ambiguous contact, sparse punctuation and missing-flank cases
word-owned. Preserve every physical point/index and paired source coordinate in
the word/rule ledger. This classification applies before immutable registration;
never rewrite accepted owners to justify a later failed correction.

Already established text owners must enter a later stock solve at zero rigid
translation. Refusing them only after solving can waste stock trials or let a
neighbour's bounding rectangle drag corrected text. Resolve a refused ordering
cycle using its actual constraint edges before changing geometry.

In an optional exact coordinate solve, protect measured pixel centers and full
half-pixel footprints under one rigid owner rather than asserting that empty
word-box corners are glyphs. Reserve the interpolation lattice halo; verify
all physical footprint points, current/retained glyphs and all established
print/normal/column/join/area gates. Empty-box bands remain the legacy default.
No later candidate may rewrite an accepted word ledger or relax a preservation
gate to accommodate its effects.

Repeated stock strengths that resolve to exactly identical owner translations
produce identical geometry within the same refinement call. After a completed
geometry refusal, retain that evidence instead of rerendering/rechecking the
identical proposal. Do not skip distinct translations or retries after an
allocation/incomplete-check failure. Every distinct candidate still receives
the full preservation proof.

A short first-cell label can contain only two connected glyph bodies and three
separated similar divider fragments. Classify these before fresh immutable
registration only with a long independently measured row, real flanks and an
observed frame crossing at any short flank. Keep exact point/index/source
accounting and all glyphs; missing joins, contact and punctuation remain
ambiguous. Compare membership distances in actual pixel units at inclusive
boundaries, rather than adding a tolerance for normalized arithmetic noise.

The source path must retain grouped native word-row corrections as well as
single-word rotations. Compose row rotations through the true retained parent,
keep all print/shape/row/column/join/continuous-area guards, and reject groups
containing established fixed text. Establish supported rule/column geometry
before fixing a full text row so a new text stage does not prematurely constrain
unfinished structural corrections. Optional experiment defaults stay unchanged
until end-to-end native release verification succeeds.

Some rigid fallback component groups can represent remnants of an already
straight crossing rule. An optional tangent-only role requires repeated thin
complete footprints, a substantial span, a measured established flat rule and
no native word owner in the group. Preserve exact physical/source points and
normal positions; native words and all other glyph groups remain rigid. Do not
grant this role to square dots, thick glyphs, short/curved/distant traces or
native-owned components. Transpose both coordinates and the normal axis through
the complete paired role ledger. Full current/retained/original glyph, column,
rule, join and continuous-area checks remain mandatory before promotion.

Defer complete pixel-footprint material inversions until an ordered candidate
reaches geometry proof; ordering refusals still use the same complete footprint
bands. A completed continuous-area refusal may be reused only for exactly
identical owner translations in the same coordinate refinement. Distinct
translations, allocation failures and incomplete proofs must still retry.

Measure the heuristic text mask in the document's reading direction before
transposing a vertical-rule solve. Transpose that complete mask together with
the raster and native owners; rebuilding the heuristic in rotated coordinates
can discard visible words omitted by native OCR and split their fallback rigid
constraints. Native OCR omissions never authorize deforming visible text.

### Virtual column origins and real ink (optional coordinate research)

Native leading-edge midpoints may be blank bearings beside a word. Track a
virtual font origin through the rigid transform of its complete immutable ink,
rather than treating the blank bearing as printed material. Recover the word
association from the original column registration frame and verify that frame
reproduces the immutable column points. Never choose a reference from the
candidate. Keep the .05px relative column limit and .1px complete word-shape
and pair-distance limits. Ambiguous, incomplete or unregistered evidence must
use the existing material-point guard. Unrelated tiny clouds do not invalidate
other complete evidence; a selected tiny cloud cannot certify a binding.

The corner dual audit proves this distinction on the pink source prefix: all
eight rejected bearing columns keep their actual font origins unchanged. It
does not establish global reproduction or change the protected production
engine. Physical column origins and tangent-only rule proxy movement are
explicit optional coordinate flags; the latter requires measured ink bands.
Every native owner, original/retained glyph, fixed text, normal rule, physical
join and continuous-area guard remains mandatory.

Local straightness alone is insufficient for row assembly. The accepted pink
header has separate straight sections at different heights around the native
title words. Detecting no bend in either section does not establish a common
straight row. Use actual separated rule sections and word evidence; do not
invent a printed bridge through letters or relax already working body guards.

A registered stroke may stop a few pixels before the currently observed ink.
A row correction must cover those real pixels too; clipping the correction to
the old registration can move the main stroke while leaving a new kink at its
end. The optional shared-row path may append only short native continuations
from a section with at least95% registered coverage and the unchanged1.5px
covered-ridge match. Preserve every original material point exactly. Refuse
word occlusion, unsafe stock, conflicting observations and unmeasured tails.
Independently reconstruct the continuation from the same native raster evidence
in the outer audit before using it for a physical join. Missing evidence keeps
the proposal refused; an endpoint clamp is never evidence of continuation.
The original glyph, current/retained shape, column, join and .2 continuous-area
limits still apply to the complete correction.

Unrecognized compact print protection (optional printed-grid research path):
Native OCR is incomplete evidence. Compact, strongly supported ink bodies that
are entirely unowned may receive conservative measured owners, with paired
immutable source/reference points. Do not reassign partial native ownership,
infer text, or absorb measured rule crossings. Nearby bodies share an owner
only with matching row/height and a short character gap. Carry and independently
audit the complete ink through later passes; the unrecognized-print shape
limit is .03 measurement pixels, stricter than the existing retained-word .1
limit. The existing tolerance cannot be raised by a caller. Preserve the paired
registry through axis transposition. Any independent original-glyph failure
rejects the candidate, even if the current native/rule guards accepted it.
Restore the last independently passing checkpoint exactly and redo the failed
pass. A checkpoint retry is not a complete original-image reproduction.

Optional retained-raster rendering:
Reuse only the exact parent's unrepaired source-map pixels outside the new
mesh region. Keep that raw raster separate from its completed protected raster;
stock estimation must never consume already repaired parent pixels. Reset
active samples to transparent before testing source availability. Preserve the
existing alpha/outside-region merge after the same stock repair. Source decode
may be shared within this one render, but never replace material coordinates
or interpolate a sampled composite mesh. Enable only after raw and repaired
native outputs match the protected result exactly. Cached render qualification
is separate from original-image quality and product end-to-end timing.

Optional legacy missed-print protection uses the same measured compact ink
owners as coordinate refinement. Keep immutable source/reference pairs for
native and supplemental owners across later OCR requests; compare supplemental
shape to its first paired geometry at the existing0.03px measurement limit.
The coordinate mode still requires its partitioned word and measured-band
prerequisites. Legacy defaults remain unchanged. A word need not be recognised
for its measured ink to prevent a correction boundary from cutting through it.
Do not promote a locally repaired pass until original-source reproduction and
previously working page geometry also pass. No new dependency or model rights.

Optional coordinate text refinement uses the same measured glyph eligibility
as its rigid word solver. A supported repeated baseline may qualify even when
varying cap heights cannot supply the separate body-center proof. Unsupported
baselines still need the existing independent body-profile fallback. This only
makes preflight consistent with refinement; all immutable shape, rule, row,
column, join and continuous-area guards remain unchanged.

Optional coordinate word-flow refinement protects neighboring words by their
complete measured ink and subpixel perimeter. Their empty rectangular corners
may remain usable stock. Rigid word, paired unknown-print, complete rule, row,
column, join and continuous-area acceptance checks still decide every proposal;
a failed proposal retains the preceding checkpoint. This option does not alter
the application's default engine until full release verification passes.


### Complete printed stroke protection during optional text correction

A ruler center is insufficient protection for a blurred printed line. Optional
coordinate word and word-row repairs now retain the complete measured contrast
ink connected to every observed ruler center, with independent word footprints
excluded. Both position axes stay fixed during a text-only repair. Separately
authorized tangent movement belongs to rule repair. Blank or unsafe samples
cannot introduce a stroke owner, and transposed rasters use contiguous scratch
masks. The existing word, saved-source, column, join and continuous area limits
remain unchanged. Native release qualification is still required before this
optional coordinate path becomes the shipping default.


### Unordered ink owners never become connected traces

Text-only correction must distinguish an ordered observed rule trace from an
unordered ink cloud. Rasterize a complete measured stroke or saved text cloud
as its actual points and pixel bodies, never as a polyline joining consecutive
samples. A row-major cloud may otherwise invent chords through unprinted gaps
or neighboring words. The same ownership distinction must survive direct
blending and bounded flow. Every original ink sample remains in the unchanged
final position/shape audits; this change removes nonexistent geometry, not a
printed constraint. Continuous source-area and retained-source limits remain
unchanged. A synthetic two-stroke cloud verifies that its invented ladder
cannot block the intervening word while both actual strokes stay fixed.

### Check original material before the first content warp

Capturing text only after a broad content warp can silently make its damage
the reference for later preservation checks. Compare the same immutable source
ink against its original perspective-only geometry from the first correction
onward. Do not interpret a later checkpoint-relative pass as proof that the
first correction preserved the original text.

The optional original-plane printed-grid path skips the coarse content warp
and unguarded preparation. It captures native words and conservatively measured
missed print before proposing any local correction. Measured ink protection and
missed-print protection are mandatory for this option; their existing shape,
source-area, join, and column limits remain unchanged. Non-grid pages keep the
existing fallback, and the application default remains unchanged until original
source rendering and previously accepted page checks pass. Never extend a
printed border from an unconfirmed faint trace.


### User quality target (2026-09-28)

The user accepts practical97% straightness and asks not to spend further work
seeking100% after visual approval. This is a user quality target, not a computed
accuracy score. Preserve the exact approved preview and advance in order:
image1 (accepted), image2, pink17, then the previously planned cases. Every
rendered result must be shown whether passing or failing. Keep original printed
material, continuous-area, identity-card and passport protection unchanged;
this visual target does not relax preservation thresholds or authorize cuts.
