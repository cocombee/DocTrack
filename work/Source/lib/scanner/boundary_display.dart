import 'package:tracker_engine/tracker_engine.dart';

/// Source-pixel inset shared with the engine, including frame-clipping rules.
List<TrackerPoint> insetBoundary(BoundaryResult result, {double pixels = 4}) {
  final segments=offsetBoundarySegments(result.boundarySegments,result.width,result.height,pixels);
  return [for(final s in segments)for(var i=0;i<(s.isCurve?32:1);i++)s.at(i/(s.isCurve?32:1))];
}
