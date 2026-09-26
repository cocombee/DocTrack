import 'dart:math' as math;

import 'package:tracker_engine/tracker_engine.dart';

import 'document_format.dart';
import 'flatten_geometry.dart';
import 'rounded_outline.dart';

/// A ratio-only estimate cannot disable paper's interior correction.
/// Confirmed identity/passport evidence retains the established rigid path.
bool isRigidDocumentForFlattening(
  BoundaryResult boundary,
  DocumentFormat format,
  List<String> textLines,
) {
  if (format.label.startsWith('ID /') ||
      format.label.startsWith('Passport') ||
      DocumentContentEvidence.fromLines(textLines).kind ==
          DocumentKind.identityDocument) {
    return true;
  }
  if (!format.label.startsWith('Card proportions')) return false;

  final geometry = FlattenGeometry(
    boundary.corners ?? [],
    boundary.boundarySegments,
    boundary.width,
    boundary.height,
    followCurves: false,
  );
  // RoundedOutline measures short corner arcs in the document plane. Long
  // bowed paper sides are not rounded-corner evidence.
  if (RoundedOutline(
        geometry,
        boundary.width,
        boundary.height,
      ).corners.length >=
      2) {
    return true;
  }
  final transitions = <int>{};
  for (final segment in boundary.boundarySegments) {
    if (segment.isCurve || segment.isImageClip) continue;
    final a = geometry.rectify(segment.start);
    final b = geometry.rectify(segment.end);
    final dx = (a.x - b.x).abs(), dy = (a.y - b.y).abs();
    // A two-point corner crosses both adjoining sides. A short piece along
    // one side, or two pieces at the same corner, cannot establish a family.
    if (dx <= .001 || dy <= .001 || math.max(dx, dy) > .2) continue;
    final midpoint = geometry.rectify(segment.at(.5));
    final cx = midpoint.x < .5 ? 0.0 : 1.0;
    final cy = midpoint.y < .5 ? 0.0 : 1.0;
    if (math.max((a.x - cx).abs(), (b.x - cx).abs()) > .2 ||
        math.max((a.y - cy).abs(), (b.y - cy).abs()) > .2) {
      continue;
    }
    transitions.add((cy == 0 ? 0 : 2) + (cx == 0 ? 0 : 1));
  }
  return transitions.length >= 2;
}
