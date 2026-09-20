# DocTrack Document Outline Rules

- Prefer the main physical paper silhouette with a small, practical number of anchors (as few as the actual shape allows; no fixed maximum), adding points only for meaningful bends.
- Preserve perspective, broad folds, rounded corners and significant missing sections. Use straight segments where appropriate and gentle Bézier curves where the paper bends.
- Do not trace speckle, micro-jaggedness, or tiny central tears/nicks. The user explicitly rejected the dense 63-point perimeter in favour of the earlier simple outline.
- Infer the paper edge behind hands, pens, staples and other occluders; these must not create dents in the paper outline.
- Current approved inset preference: an additional 2 source pixels inward from the earlier approximately 2px outline, giving approximately 4px overall. Apply consistently to every edge.
- Precision means reviewing the simplified silhouette and inward margin along the whole perimeter, not maximizing point count. Do not claim exact segmentation where the image is ambiguous.
- Images 1, 2 and 6 have been updated in work/Data. Image 1 uses 8 points; image 2 uses 9; image 6 now uses 14 after rejection of the over-smoothed 9-point outline. Image 2's simple method was accepted by the user. Preserve backups in work/Outline Review.
- Normal mouse drag box-selects anchors. The top-right Bézier icon reveals handles for all selected anchors without modifying shape. Handle rotation/length edits curvature; retain undo.
- Working source: work/Source. Installed app: /Applications/DocTrack.app. Data is now isolated in this task's work/Data; do not modify Hermes labels for subsequent edits.
- Retain Safe Build and New Build under work. outputs/DocTrack.app links to New Build. Do not promote a candidate to Safe Build without explicit build approval.
- Task metadata limitation remains: this existing task belongs to Apps but its live cwd still points to the Apps project root. No supported path-mutation tool was available; Codex UI control was blocked. Do not claim this metadata is fixed.

- Fold correction: all five visible folds in image 6 create meaningful edge-direction changes and protrusions. Preserve these with independent incoming/outgoing handles and extra anchors. The preference for fewer points never permits flattening fold geometry. The 14-point revision is pending user review, not approved.

- Rounded corners must use genuine Bézier arcs, not angular polygon approximations. Keep real crease corners sharp with independent handles.
- Remaining 47 images received sparse outline annotations and visual contact-sheet review. Occluded and clipped edges are recorded as estimates in outline_review; these annotations are drafts pending user review, not evidence of a trained detector.

## Precision correction from four user examples

The user rejected batch examples 12 (passport), 22 (BBQ receipt), 30 (German card), and 5 (sheet on keyboard). The previous contact-sheet review did not establish precise edge alignment. These remain rejected drafts until corrected and inspected closely.

- Locate the physical boundary at source resolution before applying the approximately 4-source-pixel inward offset. Do not offset a rough quadrilateral and call it precise. Screen/preview pixels are not source pixels.
- Inspect the entire perimeter at close zoom, including edge midpoints and every corner. Check that the inset stays consistently inside the real boundary; a good overall silhouette or valid non-intersecting polygon is insufficient.
- Fit each corner's actual curvature and tangent directions independently. Do not assume equal radii, square passport top corners, or a generic rounded rectangle. Join arcs to adjacent edges without kinks.
- A nominally rectangular sheet can have bowed sides, top, and bottom. Example 5 visibly requires curved edge fitting, not four straight chords. Four anchors may still carry curved segments through non-collinear Bézier handles; add anchors only when one cubic cannot follow the real bend.
- Example 22 shows uneven alignment along the receipt sides. Match visible edge sections, infer only the hand-covered intervals, and preserve smooth continuation without tracing fingers or letting the whole edge drift inward.
- Examples 12 and 30 require closer boundary and corner fitting; assigning a document category is only a shape prior, not a substitute for observed geometry.
- Preserve meaningful fold outdents while simplifying tiny tears and pixel noise. Minimize anchors only after the contour fits. Speed must come from batching work, never from skipping full-perimeter verification.
- Saving rules and annotations does not train model weights. Do not claim that the app's detector has learned these corrections unless an actual training/update pipeline has run and been evaluated.

- Image 4 correction: the black rectangular frame is printed inside the paper. Inset from the outer physical paper edge, never from the printed frame. Distinguish printing, paper, transparent laminate, and cast shadows before edge fitting; high contrast alone cannot identify a boundary.

- User approved the displayed image 44 occlusion correction after the 43/44 revision. Infer hidden corners from the continuation/intersection of visible physical edges, not where paper reappears beside the thumb. Image 43 was revised by the same method; do not infer separate explicit approval of unseen images.
- Installed v1.0.2 build 4: zoom-independent outline/handle/marker sizes and hit regions; equal 36px floating buttons; transparent hidden native title strip with 40px sidebar top padding reserving window-control space. Seven tests pass, including raster-size checks at 1x/2x/6x; flutter analyze clean.

## Current decision contract

[Engine Decision Rules](Engine%20Decision%20Rules.md) is authoritative for subsequent engine work. In particular, camera-frame closures receive no frame-normal inset; colour alone does not distinguish paper from objects; shape, folds and occlusion are assessed jointly. Historical image revisions above are not evidence that the automatic detector has passed.
