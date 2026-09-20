import 'dart:math' as math;

import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';

/// Remove only a narrow, continuously supported source-background fringe.
/// Material is measured on both sides before isolation loses exterior evidence.
Set<int> sourceBackgroundFringe(
  img.Image image,
  List<BoundarySegment> segments, {
  bool texturedExterior = false,
  bool coherentEdges = false,
}) {
  final w = image.width, h = image.height;
  final rejected = <int>{};
  List<double> pixel(double x, double y) {
    final ix = x.floor(), iy = y.floor(), fx = x - ix, fy = y - iy;
    final a = image.getPixel(ix, iy), b = image.getPixel(ix + 1, iy);
    final c = image.getPixel(ix, iy + 1), d = image.getPixel(ix + 1, iy + 1);
    return [
      for (var k = 0; k < 3; k++)
        a[k] * (1 - fx) * (1 - fy) +
            b[k] * fx * (1 - fy) +
            c[k] * (1 - fx) * fy +
            d[k] * fx * fy,
    ];
  }

  for (final segment in segments) {
    if (segment.isImageClip) continue;
    final dx = (segment.end.x - segment.start.x) * w;
    final dy = (segment.end.y - segment.start.y) * h;
    final count = math.max(12, math.sqrt(dx * dx + dy * dy).round());
    final points = [
      for (var i = 0; i < count; i++) segment.at(.005 + .99 * i / (count - 1)),
    ];
    final profiles =
        <
          ({
            double x,
            double y,
            double nx,
            double ny,
            List<double> outside,
            List<double> vector,
            double length,
            double edge,
            bool safe,
          })
        >[];
    for (var i = 0; i < count; i++) {
      final before = points[math.max(0, i - 1)],
          after = points[math.min(count - 1, i + 1)];
      final tx = (after.x - before.x) * w, ty = (after.y - before.y) * h;
      final norm = math.sqrt(tx * tx + ty * ty);
      final nx = norm > 1e-8 ? -ty / norm : 0.0,
          ny = norm > 1e-8 ? tx / norm : 0.0;
      // Tracker points use pixel boundaries; image samples use pixel centres.
      final x = points[i].x * w - .5, y = points[i].y * h - .5;
      var safe =
          norm > 1e-8 &&
          x - 6 * nx.abs() >= 0 &&
          y - 6 * ny.abs() >= 0 &&
          x + 6 * nx.abs() < w - 1 &&
          y + 6 * ny.abs() < h - 1;
      var outside = <double>[0, 0, 0], vector = <double>[0, 0, 0], length = 0.0;
      var edge = 0.0;
      if (safe) {
        final values = [
          for (var j = 0; j < 49; j++)
            pixel(x + nx * (-6 + j * .25), y + ny * (-6 + j * .25)),
        ];
        final inside = <double>[0, 0, 0];
        for (var k = 0; k < 3; k++) {
          outside[k] =
              values.take(12).map((v) => v[k]).reduce((a, b) => a + b) / 12;
          inside[k] =
              values.skip(37).map((v) => v[k]).reduce((a, b) => a + b) / 12;
          for (final pair in [
            if (!texturedExterior) (values.take(12), outside[k]),
            (values.skip(37), inside[k]),
          ]) {
            final variance =
                pair.$1
                    .map((v) => math.pow(v[k] - pair.$2, 2).toDouble())
                    .reduce((a, b) => a + b) /
                12;
            if (variance >= 144) safe = false;
          }
          vector[k] = inside[k] - outside[k];
          length += vector[k] * vector[k];
        }
        var best = -1.0;
        for (var j = 0; j < 47; j++) {
          var strength = 0.0;
          for (var k = 0; k < 3; k++) {
            final v = values[j + 2][k] - values[j][k];
            strength += v * v;
          }
          if (strength > best) {
            best = strength;
            edge = -6 + (j + 1) * .25;
          }
        }
        safe =
            safe &&
            length > 5625 &&
            edge > (coherentEdges ? -.75 : .25) &&
            edge < 4.5;
      }
      profiles.add((
        x: x,
        y: y,
        nx: nx,
        ny: ny,
        outside: outside,
        vector: vector,
        length: length,
        edge: edge,
        safe: safe,
      ));
    }
    for (var i = 0; i < count; i++) {
      final p = profiles[i];
      if (!p.safe) continue;
      // A short ink mark shifts the transition inward only locally. Require
      // neighbours to agree on its position before treating it as background.
      var support = 0;
      for (var j = math.max(0, i - 5); j <= math.min(count - 1, i + 5); j++) {
        if (profiles[j].safe &&
            (!coherentEdges || (profiles[j].edge - p.edge).abs() < 1.25)) {
          support++;
        }
      }
      if (support < 8) continue;
      for (
        var y = math.max(0, p.y.round() - 5);
        y < math.min(h, p.y.round() + 6);
        y++
      ) {
        for (
          var x = math.max(0, p.x.round() - 5);
          x < math.min(w, p.x.round() + 6);
          x++
        ) {
          final dx = x - p.x,
              dy = y - p.y,
              inset = dx * p.nx + dy * p.ny,
              along = (p.nx * dy - p.ny * dx).abs();
          if (inset < -.5 || inset > 4.5 || along > .8) continue;
          final color = image.getPixel(x, y);
          var coverage = 0.0;
          for (var k = 0; k < 3; k++) {
            coverage += (color[k] - p.outside[k]) * p.vector[k];
          }
          coverage /= p.length;
          var residual = 0.0;
          for (var k = 0; k < 3; k++) {
            final d = color[k] - p.outside[k] - coverage * p.vector[k];
            residual += d * d;
          }
          if (coverage < .85 && residual < 400) rejected.add(y * w + x);
        }
      }
    }
  }
  return rejected;
}
