import 'dart:math' as math;

import 'flatten_geometry.dart';

/// Corner radii are measured in output pixels, after perspective correction.
class RoundedOutline {
  RoundedOutline(
    FlattenGeometry geometry,
    this.width,
    this.height, {
    bool normalizeIdCorners = false,
    bool repairMissingCorners = true,
  }) {
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
    // A low-contrast or occluded rounded corner is sometimes simplified by
    // boundary fitting into one short straight chamfer. If another corner
    // already establishes a rounded family, treat only these short corner
    // transitions as the matching rounded corner. A hard passport corner is
    // made from two long sides and therefore does not qualify here.
    var matchingShortCorners = 0;
    if ((repairMissingCorners && corners.length >= 2) || normalizeIdCorners) {
      for (var i = 0; i < geometry.segments.length; i++) {
        final s = geometry.segments[i];
        if (s.isCurve || s.isImageClip) continue;
        final a = geometry.rectify(s.start), b = geometry.rectify(s.end);
        final length = math.sqrt(
          math.pow((a.x - b.x) * width, 2) +
              math.pow((a.y - b.y) * height, 2),
        );
        if (length <= 2 || length > math.min(width, height) * .2) continue;
        final mid = geometry.rectify(s.at(.5));
        final distances = [
          math.sqrt(mid.x * mid.x + mid.y * mid.y),
          math.sqrt((mid.x - 1) * (mid.x - 1) + mid.y * mid.y),
          math.sqrt(mid.x * mid.x + (mid.y - 1) * (mid.y - 1)),
          math.sqrt((mid.x - 1) * (mid.x - 1) + (mid.y - 1) * (mid.y - 1)),
        ];
        final corner = distances.indexOf(distances.reduce(math.min));
        if (distances[corner] < .22) {
          matchingShortCorners++;
          corners.add(corner);
        }
      }
    }
    // Two clear arcs plus one matching short transition are sufficient to
    // establish the rounded stock family; the remaining corner is commonly
    // hidden by glare, an emblem, or a low-contrast edge.
    if (repairMissingCorners && !normalizeIdCorners && corners.length >= 2 && matchingShortCorners > 0) {
      corners.addAll([0, 1, 2, 3]);
    }
    if (corners.length < 2) {
      corners.clear(); // One uncertain arc cannot establish a corner family.
      return;
    }
    if (repairMissingCorners && !normalizeIdCorners && corners.length == 3) {
      corners.addAll([0, 1, 2, 3]);
    }
    measurements.sort();
    radius = measurements[measurements.length ~/ 2].clamp(
      0,
      math.min(width, height) * .2,
    );
    if (normalizeIdCorners) {
      // ID cards use one common physical corner radius. A glare, crop, or
      // weak edge must not make one corner visibly smaller than the others.
      corners.addAll([0, 1, 2, 3]);
      radius = math.max(radius, math.min(width, height) * .06);
    }
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
