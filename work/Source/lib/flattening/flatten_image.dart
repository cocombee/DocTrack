import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';

import 'flatten_geometry.dart';
import 'background_surface.dart';
import 'rounded_outline.dart';
import 'document_format.dart';
import 'interior_mesh.dart';

class FlattenedImage {
  const FlattenedImage(
    this.png,
    this.width,
    this.height,
    this.ratio,
    this.format,
  );
  final Uint8List png;
  final int width, height;
  final double ratio;
  final String format;
}

Future<FlattenedImage> flattenImage(
  Uint8List bytes,
  BoundaryResult boundary, {
  double? ratio,
  bool followCurves = true,
  int maxDimension = 1600,
  List<String> textLines = const [],
  double? estimatedRatio,
  InteriorMesh? interiorMesh,
  bool repairBorder = true,
  bool? preserveRoundedCorners,
}) => Isolate.run(() {
  if (maxDimension < 16 || maxDimension > 4096) {
    throw ArgumentError('Invalid output resolution');
  }
  final decoded = img.decodeImage(bytes);
  if (decoded == null) {
    throw const FormatException('Image could not be decoded');
  }
  final source = img.bakeOrientation(decoded);
  if (source.width != boundary.width || source.height != boundary.height) {
    throw ArgumentError('The image and detected shape do not match');
  }
  final map = FlattenGeometry(
    boundary.corners ?? [],
    boundary.boundarySegments,
    source.width,
    source.height,
    // Perspective establishes the document plane first. Once an accepted
    // interior mesh exists, printed text/rules/shapes are the surface ruler;
    // do not let the crop edge bend the content a second time.
    // Interior mesh correction must not disable the measured outer boundary
    // curves. Rounded card corners are boundary geometry, independent from
    // the content mesh used inside the document.
    followCurves: followCurves,
  );
  final format = identifyDocumentFormat(
    estimatedRatio ?? map.estimatedRatio,
    textLines,
  );
  final cardFamily = format.label.startsWith('ID /') ||
      format.label.startsWith('Passport') ||
      format.label.startsWith('Business card') ||
      format.label.startsWith('Credit card');
  // Paper keeps the measured crop ratio. Known card families may use their
  // standard physical ratio, but paper must never be stretched to fill a
  // guessed Letter/A4/B5 frame; missing areas are repaired later.
  final aspect =
      ratio ??
      (cardFamily
          ? format.orientedRatio(estimatedRatio ?? map.estimatedRatio)
          : null) ??
      estimatedRatio ??
      map.estimatedRatio;
  if (!aspect.isFinite || aspect < .1 || aspect > 10) {
    throw ArgumentError('Invalid proportions');
  }
  final content = DocumentContentEvidence.fromLines(textLines);
  final w = aspect >= 1 ? maxDimension : (maxDimension * aspect).round();
  final h = aspect >= 1 ? (maxDimension / aspect).round() : maxDimension;
  final outline = RoundedOutline(map, w, h);
  // Size recognition cannot discard an accepted two/four-corner family.
  // Transaction paper still repairs tears and folds to a rectangular sheet.
  final rounded =
      preserveRoundedCorners ??
      preserveDocumentOutline(
        format,
        content,
        hasRoundedCorners: outline.corners.length >= 2,
      );
  bool insideShape(double u, double v) => !rounded || outline.contains(u, v);
  final coverageRepair =
      repairBorder &&
      !rounded &&
      const bool.fromEnvironment('DOCTRACK_COVERAGE_REPAIR');
  final output = img.Image(width: w, height: h, numChannels: 4);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final u = (x + .5) / w, v = (y + .5) / h;
      if (repairBorder && !insideShape(u, v)) continue;
      final local = interiorMesh?.at(u, v) ?? TrackerPoint(u, v);
      final p = map.source(local.x, local.y);
      final sx = p.x * source.width - .5, sy = p.y * source.height - .5;
      final available =
          sx >= 0 &&
          sy >= 0 &&
          sx <= source.width - 1 &&
          sy <= source.height - 1 &&
          map.contains(p) &&
          (coverageRepair || source.getPixel(sx.round(), sy.round()).a > 0);
      if (!available) {
        continue;
      }
      final ix = sx.floor(), iy = sy.floor(), fx = sx - ix, fy = sy - iy;
      final a = source.getPixel(ix, iy),
          b = source.getPixel(math.min(ix + 1, source.width - 1), iy);
      final c = source.getPixel(ix, math.min(iy + 1, source.height - 1)),
          d = source.getPixel(
            math.min(ix + 1, source.width - 1),
            math.min(iy + 1, source.height - 1),
          );
      final weights = [
        (1 - fx) * (1 - fy) * a.a / 255,
        fx * (1 - fy) * b.a / 255,
        (1 - fx) * fy * c.a / 255,
        fx * fy * d.a / 255,
      ];
      final weight = weights.reduce((a, b) => a + b);
      if (weight <= 0) continue;
      double mix(num a, num b, num c, num d) =>
          (a * weights[0] + b * weights[1] + c * weights[2] + d * weights[3]) /
          weight;
      output.setPixelRgba(
        x,
        y,
        mix(a.r, b.r, c.r, d.r),
        mix(a.g, b.g, c.g, d.g),
        mix(a.b, b.b, c.b, d.b),
        coverageRepair ? (weight * 255).round() : 255,
      );
    }
  }
  if (repairBorder) {
    final surface = BackgroundSurface.estimate(
      output,
      localFallback: const bool.fromEnvironment(
        'DOCTRACK_LOCAL_STOCK_REPAIR',
        defaultValue: true,
      ),
    );
    final repaired = <int>[];
    final partial = <int, List<double>>{};
    if (!rounded) {
      for (final index in {
        ...exteriorStrips(output, surface),
        ...missingEdgeFringe(output, surface),
        if (const bool.fromEnvironment('DOCTRACK_WIDE_FRINGE'))
          ...missingEdgeFringe(
            output,
            surface,
            depthFraction: .06,
            darkOnly: true,
          ),
      }) {
        output.setPixelRgba(index % w, index ~/ w, 0, 0, 0, 0);
      }
    }
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final u = (x + .5) / w, v = (y + .5) / h;
        final pixel = output.getPixel(x, y);
        if ((coverageRepair ? pixel.a == 255 : pixel.a != 0) ||
            !insideShape(u, v)) {
          continue;
        }
        if (pixel.a > 0) {
          partial[y * w + x] = [
            pixel.r.toDouble(),
            pixel.g.toDouble(),
            pixel.b.toDouble(),
            pixel.a / 255,
          ];
        }
        repaired.add(y * w + x);
        final color = surface.color(u, v);
        output.setPixelRgba(x, y, color[0], color[1], color[2], 255);
      }
    }
    const adjacentStock = bool.fromEnvironment('DOCTRACK_ADJACENT_STOCK');
    if (adjacentStock) extendLocalStock(output, repaired, surface);
    for (final entry in partial.entries) {
      final x = entry.key % w, y = entry.key ~/ w;
      final original = entry.value, stock = output.getPixel(x, y);
      final a = original[3];
      output.setPixelRgba(
        x,
        y,
        original[0] * a + stock.r * (1 - a),
        original[1] * a + stock.g * (1 - a),
        original[2] * a + stock.b * (1 - a),
        255,
      );
    }
    featherBackground(output, repaired, surface, adjacentColor: adjacentStock);
  }
  return FlattenedImage(
    Uint8List.fromList(img.encodePng(output)),
    w,
    h,
    aspect,
    ratio == null ? format.label : 'Custom proportions',
  );
});
