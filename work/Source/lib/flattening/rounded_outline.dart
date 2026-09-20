import 'dart:math' as math;

import 'flatten_geometry.dart';

/// Corner radii are measured in output pixels, after perspective correction.
class RoundedOutline {
  RoundedOutline(FlattenGeometry geometry, this.width, this.height) {
    final measurements = <double>[];
    for (final s in geometry.segments) {
      if (!s.isCurve) continue;
      final a = geometry.rectify(s.start), b = geometry.rectify(s.end);
      final m = geometry.rectify(s.at(.5));
      if (math.max((a.x - b.x).abs(), (a.y - b.y).abs()) > .2) continue;
      final cx = m.x < .5 ? 0.0 : 1.0, cy = m.y < .5 ? 0.0 : 1.0;
      final rx = math.max((a.x - cx).abs(), (b.x - cx).abs());
      final ry = math.max((a.y - cy).abs(), (b.y - cy).abs());
      if (rx <= .001 || ry <= .001 || rx > .2 || ry > .2) continue;
      corners.add((cy == 0 ? 0 : 2) + (cx == 0 ? 0 : 1));
      measurements.add((rx * width + ry * height) / 2);
    }
    if (corners.length < 2) {
      corners.clear(); // One uncertain arc cannot establish a corner family.
      return;
    }
    if (corners.length == 3) corners.addAll([0, 1, 2, 3]);
    measurements.sort();
    radius = measurements[measurements.length ~/ 2].clamp(
      0,
      math.min(width, height) * .2,
    );
  }
  final int width, height;
  final Set<int> corners = {};
  double radius = 0;
  bool contains(double u, double v) {
    final x = u * width, y = v * height;
    for (final corner in corners) {
      final dx = corner % 2 == 0 ? x : width - x;
      final dy = corner < 2 ? y : height - y;
      if (dx < radius &&
          dy < radius &&
          math.pow(radius - dx, 2) + math.pow(radius - dy, 2) >
              radius * radius) {
        return false;
      }
    }
    return true;
  }
}
