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

class _MeshRaster {
  img.Image? raw;
  img.Image? completed;
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
  double documentScale = 1.0,
  bool reuseParentRaster = const bool.fromEnvironment(
    'DOCTRACK_REUSE_PARENT_RASTER',
  ),
}) => Isolate.run(
  () => _flattenImage(
    bytes,
    boundary,
    ratio: ratio,
    followCurves: followCurves,
    maxDimension: maxDimension,
    textLines: textLines,
    estimatedRatio: estimatedRatio,
    interiorMesh: interiorMesh,
    repairBorder: repairBorder,
    preserveRoundedCorners: preserveRoundedCorners,
    documentScale: documentScale,
    reuseParentRaster: reuseParentRaster,
  ),
);

FlattenedImage _flattenImage(
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
  double documentScale = 1.0,
  bool reuseParentRaster = false,
  img.Image? decodedSource,
  _MeshRaster? capture,
  bool encode = true,
}) {
  if (maxDimension < 16 || maxDimension > 4096) {
    throw ArgumentError('Invalid output resolution');
  }
  if (!documentScale.isFinite || documentScale < 1.0 || documentScale > 1.25) {
    throw ArgumentError('Invalid document scale');
  }
  final decoded = decodedSource ?? img.decodeImage(bytes);
  if (decoded == null) {
    throw const FormatException('Image could not be decoded');
  }
  final source = decodedSource ?? img.bakeOrientation(decoded);
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
  final standardCardFamily =
      format.label.startsWith('ID /') ||
      format.label.startsWith('Passport') ||
      format.label.startsWith('Business card') ||
      format.label.startsWith('Credit card');
  // Paper keeps the measured crop ratio. Known card families may use their
  // standard physical ratio, but paper must never be stretched to fill a
  // guessed Letter/A4/B5 frame; missing areas are repaired later.
  final aspect =
      ratio ??
      (standardCardFamily
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
  final outline = RoundedOutline(
    map,
    w,
    h,
    normalizeIdCorners:
        !format.label.startsWith('Passport') &&
        (format.label.startsWith('ID /') ||
            content.kind == DocumentKind.identityDocument),
    repairMissingCorners: !format.label.startsWith('Passport'),
  );
  // Size recognition cannot discard an accepted two/four-corner family.
  // Transaction paper still repairs tears and folds to a rectangular sheet.
  final idFamily =
      format.label.startsWith('ID /') ||
      content.kind == DocumentKind.identityDocument;
  // Once detection identifies an ID family, its rounded card clip is absolute;
  // later border/line evidence cannot downgrade it to a hard rectangle.
  final rounded = idFamily
      ? true
      : (preserveRoundedCorners ??
            preserveDocumentOutline(
              format,
              content,
              hasRoundedCorners: outline.corners.length >= 2,
            ));
  // A confirmed rounded outline is already the document silhouette. Do not
  // synthesize stock/background pixels along that edge.
  final roundedCornerCount = outline.corners.length;
  var shortCornerCount = 0;
  for (final s in map.segments) {
    if (s.isCurve || s.isImageClip) continue;
    final a = map.rectify(s.start), b = map.rectify(s.end);
    final length = math.sqrt(
      math.pow((a.x - b.x) * w, 2) + math.pow((a.y - b.y) * h, 2),
    );
    if (length <= math.min(w, h) * .2) shortCornerCount++;
  }
  final hardCornerEdgeCount = map.segments
      .where((s) => !s.isCurve && !s.isImageClip)
      .length;
  // Familiar card/passport proportions become a card-family decision only
  // when the boundary also supplies corner evidence: two rounded corners, or
  // two explicit hard corner transitions. This keeps ordinary paper on the
  // stock edge-repair path.
  final cardFamily =
      (format.label.startsWith('ID /') ||
          format.label.startsWith('Passport')) &&
      (roundedCornerCount >= 2 || shortCornerCount >= 2) &&
      hardCornerEdgeCount <= 2;
  final doRepairBorder = repairBorder && !rounded && !cardFamily;
  bool insideShape(double u, double v) => !rounded || outline.contains(u, v);
  final coverageRepair =
      repairBorder &&
      !rounded &&
      const bool.fromEnvironment('DOCTRACK_COVERAGE_REPAIR');
  final preservedMesh = interiorMesh?.preservedMesh;
  _MeshRaster? parentRaster;
  if (reuseParentRaster && preservedMesh != null) {
    parentRaster = _MeshRaster();
    _flattenImage(
      bytes,
      boundary,
      ratio: ratio,
      followCurves: followCurves,
      maxDimension: maxDimension,
      textLines: textLines,
      estimatedRatio: estimatedRatio,
      interiorMesh: preservedMesh,
      repairBorder: repairBorder,
      preserveRoundedCorners: preserveRoundedCorners,
      documentScale: documentScale,
      reuseParentRaster: true,
      decodedSource: source,
      capture: parentRaster,
      encode: false,
    );
  }
  final output = parentRaster == null
      ? img.Image(width: w, height: h, numChannels: 4)
      : img.Image.from(parentRaster.raw!);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final u = (x + .5) / w, v = (y + .5) / h;
      // Rounded IDs always use the fixed silhouette as a clip. Rectangular
      // documents use the same predicate only while performing edge repair.
      if ((rounded || doRepairBorder) && !insideShape(u, v)) continue;
      // A small ID-only overscan trims uncertain detector pixels at the
      // perimeter without changing the detected quadrilateral or paper crop.
      final sampleU = ((u - .5) / documentScale + .5).clamp(0.0, 1.0);
      final sampleV = ((v - .5) / documentScale + .5).clamp(0.0, 1.0);
      if (parentRaster != null) {
        if (!interiorMesh!.refinementRegions.any(
          (r) =>
              sampleU >= r[0] &&
              sampleU <= r[2] &&
              sampleV >= r[1] &&
              sampleV <= r[3],
        ))
          continue;
        // A changed source coordinate may become unavailable. Keep the same
        // transparent starting pixel as a full redraw within this patch.
        output.setPixelRgba(x, y, 0, 0, 0, 0);
      }
      final local =
          interiorMesh?.at(sampleU, sampleV) ?? TrackerPoint(sampleU, sampleV);
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
  // Keep the unrepaired source-map raster separate from the protected result.
  // Stock estimation must see the same raw pixels as the complete renderer.
  if (capture != null) capture.raw = img.Image.from(output);
  if (doRepairBorder) {
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
  if (preservedMesh != null) {
    final preserved =
        parentRaster?.completed ??
        img.decodePng(
          _flattenImage(
            bytes,
            boundary,
            ratio: ratio,
            followCurves: followCurves,
            maxDimension: maxDimension,
            textLines: textLines,
            estimatedRatio: estimatedRatio,
            interiorMesh: preservedMesh,
            repairBorder: repairBorder,
            preserveRoundedCorners: preserveRoundedCorners,
            documentScale: documentScale,
          ).png,
        )!;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final u = (((x + .5) / w - .5) / documentScale + .5).clamp(0.0, 1.0);
        final v = (((y + .5) / h - .5) / documentScale + .5).clamp(0.0, 1.0);
        final inRefinement = interiorMesh!.refinementRegions.any(
          (r) => u >= r[0] && u <= r[2] && v >= r[1] && v <= r[3],
        );
        final previous = preserved.getPixel(x, y);
        // Edge/stock repair can otherwise alter pixels outside a local warp.
        // Reuse the accepted render there and preserve its exact silhouette.
        if (!inRefinement ||
            previous.a != 255 ||
            output.getPixel(x, y).a != 255) {
          output.setPixelRgba(
            x,
            y,
            previous.r,
            previous.g,
            previous.b,
            previous.a,
          );
        }
      }
    }
  }
  if (capture != null) capture.completed = output;
  return FlattenedImage(
    encode ? Uint8List.fromList(img.encodePng(output)) : Uint8List(0),
    w,
    h,
    aspect,
    ratio == null ? format.label : 'Custom proportions',
  );
}
