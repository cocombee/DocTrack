import 'dart:math' as math;

import 'package:tracker_engine/tracker_engine.dart';

/// Destination-to-source map. Physical geometry is independent of display inset.
class FlattenGeometry {
  FlattenGeometry(
    List<TrackerPoint> corners,
    this.segments,
    this.sourceWidth,
    this.sourceHeight, {
    this.followCurves = true,
    List<TrackerPoint>? projectionCorners,
  }) : corners = List.unmodifiable(corners),
       projectionCorners = List.unmodifiable(projectionCorners ?? corners) {
    if (corners.length != 4 || segments.isEmpty) {
      throw ArgumentError('A detected four-corner document is required.');
    }
    if (projectionCorners != null && followCurves) {
      throw ArgumentError(
        'A separate projection frame establishes perspective before dewarping.',
      );
    }
    final q = this.projectionCorners;
    if (q.length != 4)
      throw ArgumentError('Four projection corners are required');
    for (final p in corners) {
      if (!p.x.isFinite || !p.y.isFinite)
        throw ArgumentError('Invalid crop corner');
    }
    for (final p in q) {
      if (!p.x.isFinite || !p.y.isFinite) throw ArgumentError('Invalid corner');
    }
    if (projectionCorners != null) {
      for (var i = 0; i < 4; i++) {
        final a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
        if ((b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x) <= 1e-10) {
          throw ArgumentError(
            'The projection frame must be convex and ordered',
          );
        }
      }
    }
    final dx1 = q[1].x - q[2].x, dx2 = q[3].x - q[2].x;
    final dy1 = q[1].y - q[2].y, dy2 = q[3].y - q[2].y;
    final dx3 = q[0].x - q[1].x + q[2].x - q[3].x;
    final dy3 = q[0].y - q[1].y + q[2].y - q[3].y;
    final det = dx1 * dy2 - dx2 * dy1;
    if (det.abs() < 1e-10) throw ArgumentError('Degenerate document');
    final g = (dx3 * dy2 - dx2 * dy3) / det, h = (dx1 * dy3 - dx3 * dy1) / det;
    matrix = [
      q[1].x - q[0].x + g * q[1].x,
      q[3].x - q[0].x + h * q[3].x,
      q[0].x,
      q[1].y - q[0].y + g * q[1].y,
      q[3].y - q[0].y + h * q[3].y,
      q[0].y,
      g,
      h,
      1,
    ];
    inverse = _invert(matrix);
    polygon = [
      for (final s in segments)
        for (var j = 0; j < (s.isCurve ? 48 : 1); j++)
          s.at(j / (s.isCurve ? 48 : 1)),
    ];
    profiles = List.generate(4, (_) => <TrackerPoint>[]);
    // Short corner arcs remain silhouettes, never side-deformation evidence.
    for (final s in segments) {
      if (s.isImageClip) continue;
      final a = rectify(s.start), b = rectify(s.end), mid = rectify(s.at(.5));
      if (math.max((b.x - a.x).abs(), (b.y - a.y).abs()) < .25) continue;
      final distances = [
        mid.y.abs(),
        (mid.x - 1).abs(),
        (mid.y - 1).abs(),
        mid.x.abs(),
      ];
      final side = List.generate(
        4,
        (i) => i,
      ).reduce((a, b) => distances[a] < distances[b] ? a : b);
      for (var j = 0; j <= 64; j++) {
        final p = rectify(s.at(j / 64));
        final t = side.isEven ? p.x : p.y;
        final displacement = side.isEven
            ? p.y - (side == 2 ? 1 : 0)
            : p.x - (side == 1 ? 1 : 0);
        if (t >= 0 && t <= 1 && displacement.abs() < .18) {
          profiles[side].add(TrackerPoint(t, displacement));
        }
      }
    }
    for (final profile in profiles) {
      profile.addAll([const TrackerPoint(0, 0), const TrackerPoint(1, 0)]);
      profile.sort((a, b) => a.x.compareTo(b.x));
    }
    // Reject a foldover instead of sampling a mirrored or torn interior.
    for (var y = 0; y < 24; y++) {
      for (var x = 0; x < 24; x++) {
        final p = plane(x / 24, y / 24),
            a = plane((x + 1) / 24, y / 24),
            b = plane(x / 24, (y + 1) / 24);
        if ((a.x - p.x) * (b.y - p.y) - (a.y - p.y) * (b.x - p.x) <= 1e-7) {
          throw ArgumentError(
            'The estimated surface folds over. Try perspective only.',
          );
        }
      }
    }
  }

  /// First pins and segments retain the crop. The second virtual frame supplies
  /// camera geometry only; a separately measured content map dewarps afterward.
  /// Rectified crop samples may extend outside [0, 1]; the caller must retain
  /// them in its output canvas rather than clip to the virtual rectangle.
  FlattenGeometry.withProjectionFrame(
    List<TrackerPoint> cropCorners,
    List<BoundarySegment> segments,
    int sourceWidth,
    int sourceHeight, {
    required List<TrackerPoint> referenceCorners,
  }) : this(
         cropCorners,
         segments,
         sourceWidth,
         sourceHeight,
         followCurves: false,
         projectionCorners: referenceCorners,
       );

  final List<TrackerPoint> corners;
  final List<TrackerPoint> projectionCorners;
  final List<BoundarySegment> segments;
  final int sourceWidth, sourceHeight;
  final bool followCurves;
  late final List<double> matrix, inverse;
  late final List<TrackerPoint> polygon;
  late final List<List<TrackerPoint>> profiles;
  static List<double> _invert(List<double> m) {
    final a = m[0],
        b = m[1],
        c = m[2],
        d = m[3],
        e = m[4],
        f = m[5],
        g = m[6],
        h = m[7],
        i = m[8];
    final det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (det.abs() < 1e-12) throw ArgumentError('Degenerate perspective');
    return [
      (e * i - f * h) / det,
      (c * h - b * i) / det,
      (b * f - c * e) / det,
      (f * g - d * i) / det,
      (a * i - c * g) / det,
      (c * d - a * f) / det,
      (d * h - e * g) / det,
      (b * g - a * h) / det,
      (a * e - b * d) / det,
    ];
  }

  static TrackerPoint _map(List<double> m, double x, double y) {
    final z = m[6] * x + m[7] * y + m[8];
    if (z.abs() < 1e-9) throw ArgumentError('Unbounded perspective');
    return TrackerPoint(
      (m[0] * x + m[1] * y + m[2]) / z,
      (m[3] * x + m[4] * y + m[5]) / z,
    );
  }

  TrackerPoint rectify(TrackerPoint p) => _map(inverse, p.x, p.y);
  double _offset(int side, double t) {
    final p = profiles[side];
    for (var i = 1; i < p.length; i++) {
      if (t <= p[i].x) {
        final span = p[i].x - p[i - 1].x;
        if (span < 1e-9) return p[i].y;
        return p[i - 1].y + (p[i].y - p[i - 1].y) * (t - p[i - 1].x) / span;
      }
    }
    return 0;
  }

  TrackerPoint plane(double u, double v) => followCurves
      ? TrackerPoint(
          u + (1 - u) * _offset(3, v) + u * _offset(1, v),
          v + (1 - v) * _offset(0, u) + v * _offset(2, u),
        )
      : TrackerPoint(u, v);
  TrackerPoint source(double u, double v) {
    final p = plane(u, v);
    return _map(matrix, p.x, p.y);
  }

  /// Single-view metric estimate under square pixels and a centered principal
  /// point. When perspective does not constrain focal length, use edge lengths.
  double? get projectiveRatio {
    final g = matrix[6], h = matrix[7];
    if ((g * h).abs() < 1e-6) return null;
    final ax = (matrix[0] - .5 * g) * sourceWidth,
        ay = (matrix[3] - .5 * g) * sourceHeight;
    final bx = (matrix[1] - .5 * h) * sourceWidth,
        by = (matrix[4] - .5 * h) * sourceHeight;
    final f2 = -(ax * bx + ay * by) / (g * h);
    final scale = math.max(sourceWidth, sourceHeight).toDouble();
    if (f2 < scale * scale * .16 || f2 > scale * scale * 25) return null;
    final ratio = math.sqrt(
      (ax * ax + ay * ay + f2 * g * g) / (bx * bx + by * by + f2 * h * h),
    );
    return ratio.isFinite && ratio > .1 && ratio < 10 ? ratio : null;
  }

  double get estimatedRatio {
    final metric = projectiveRatio;
    if (metric != null) return metric;
    double length(int a, int b) => math.sqrt(
      math.pow(
            (projectionCorners[a].x - projectionCorners[b].x) * sourceWidth,
            2,
          ) +
          math.pow(
            (projectionCorners[a].y - projectionCorners[b].y) * sourceHeight,
            2,
          ),
    );
    return ((length(0, 1) + length(2, 3)) / (length(1, 2) + length(3, 0)))
        .clamp(.1, 10);
  }

  /// Ideal outline repairs accidental cuts but retains intentional corner arcs.
  bool idealContains(double u, double v) {
    for (final segment in segments) {
      if (!segment.isCurve) continue;
      final a = rectify(segment.start),
          b = rectify(segment.end),
          m = rectify(segment.at(.5));
      if (math.max((a.x - b.x).abs(), (a.y - b.y).abs()) > .2) continue;
      final cx = m.x < .5 ? 0.0 : 1.0, cy = m.y < .5 ? 0.0 : 1.0;
      final rx = math.max((a.x - cx).abs(), (b.x - cx).abs()),
          ry = math.max((a.y - cy).abs(), (b.y - cy).abs());
      if (rx <= .001 || ry <= .001 || rx > .2 || ry > .2) continue;
      final dx = (u - cx).abs(), dy = (v - cy).abs();
      if (dx < rx &&
          dy < ry &&
          math.pow((rx - dx) / rx, 2) + math.pow((ry - dy) / ry, 2) > 1) {
        return false;
      }
    }
    return true;
  }

  bool contains(TrackerPoint p) {
    var inside = false;
    for (var i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      final a = polygon[i], b = polygon[j];
      if ((a.y > p.y) != (b.y > p.y) &&
          p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x) {
        inside = !inside;
      }
    }
    return inside;
  }
}
