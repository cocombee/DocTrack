import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';

import 'flatten_geometry.dart';
import 'document_format.dart';
import 'rounded_outline.dart';
import 'source_matte.dart';

class IsolatedDocument {
  const IsolatedDocument(
    this.png,
    this.boundary,
    this.estimatedRatio, {
    this.measurementPng,
  });
  final Uint8List png;

  /// Stable material mask used only to measure layout, never as repaired output.
  final Uint8List? measurementPng;
  final BoundaryResult boundary;
  final double estimatedRatio;
}

/// Phase one changes only selection and framing. It performs no geometric warp.
Future<IsolatedDocument> isolateDocument(
  Uint8List bytes,
  BoundaryResult boundary, {
  List<String> textLines = const [],
  bool coherentEdges = const bool.fromEnvironment(
    'DOCTRACK_COHERENT_FRINGE',
    defaultValue: true,
  ),
}) => Isolate.run(() {
  final decoded = img.decodeImage(bytes);
  if (decoded == null) throw const FormatException('Unreadable image');
  final source = img.bakeOrientation(decoded);
  if (source.width != boundary.width || source.height != boundary.height) {
    throw ArgumentError('Image geometry mismatch');
  }
  final geometry = FlattenGeometry(
    boundary.corners!,
    boundary.boundarySegments,
    source.width,
    source.height,
    followCurves: false,
  );
  final preserve = preserveDocumentOutline(
    identifyDocumentFormat(geometry.estimatedRatio, textLines),
    DocumentContentEvidence.fromLines(textLines),
    hasRoundedCorners:
        RoundedOutline(geometry, source.width, source.height).corners.length >=
        2,
  );
  final fringe =
      const bool.fromEnvironment('DOCTRACK_SOURCE_MATTE', defaultValue: true) &&
          !preserve
      ? sourceBackgroundFringe(
          source,
          boundary.boundarySegments,
          coherentEdges: coherentEdges,
          texturedExterior: const bool.fromEnvironment(
            'DOCTRACK_TEXTURED_EXTERIOR',
          ),
        )
      : <int>{};
  final measurementFringe =
      coherentEdges &&
          !preserve &&
          const bool.fromEnvironment(
            'DOCTRACK_SOURCE_MATTE',
            defaultValue: true,
          )
      ? sourceBackgroundFringe(
          source,
          boundary.boundarySegments,
          texturedExterior: const bool.fromEnvironment(
            'DOCTRACK_TEXTURED_EXTERIOR',
          ),
        )
      : null;
  final polygon = [
    for (final s in boundary.boundarySegments)
      for (var j = 0; j < (s.isCurve ? 64 : 1); j++)
        s.at(j / (s.isCurve ? 64 : 1)),
  ];
  if (polygon.length < 3) {
    throw ArgumentError('A selected document is required');
  }
  final left = (polygon.map((p) => p.x).reduce(math.min) * source.width)
      .floor()
      .clamp(0, source.width - 1);
  final top = (polygon.map((p) => p.y).reduce(math.min) * source.height)
      .floor()
      .clamp(0, source.height - 1);
  final right = (polygon.map((p) => p.x).reduce(math.max) * source.width)
      .ceil()
      .clamp(left + 1, source.width);
  final bottom = (polygon.map((p) => p.y).reduce(math.max) * source.height)
      .ceil()
      .clamp(top + 1, source.height);
  final output = img.Image(
    width: right - left,
    height: bottom - top,
    numChannels: 4,
  );
  final measurement = measurementFringe == null
      ? null
      : img.Image(width: output.width, height: output.height, numChannels: 4);
  for (var y = top; y < bottom; y++) {
    final line = (y + .5) / source.height;
    final crossings = <double>[];
    for (var i = 0; i < polygon.length; i++) {
      final a = polygon[i], b = polygon[(i + 1) % polygon.length];
      if ((a.y > line) != (b.y > line)) {
        crossings.add(
          (a.x + (line - a.y) * (b.x - a.x) / (b.y - a.y)) * source.width,
        );
      }
    }
    crossings.sort();
    for (var i = 0; i + 1 < crossings.length; i += 2) {
      final x0 = (crossings[i] - .5).ceil().clamp(left, right),
          x1 = (crossings[i + 1] - .5).ceil().clamp(left, right);
      for (var x = x0; x < x1; x++) {
        final index = y * source.width + x;
        final p = source.getPixel(x, y);
        if (!fringe.contains(index)) {
          output.setPixelRgba(x - left, y - top, p.r, p.g, p.b, p.a);
        }
        if (measurement != null && !measurementFringe!.contains(index)) {
          measurement.setPixelRgba(x - left, y - top, p.r, p.g, p.b, p.a);
        }
      }
    }
  }
  TrackerPoint move(TrackerPoint p) => TrackerPoint(
    (p.x * source.width - left) / output.width,
    (p.y * source.height - top) / output.height,
  );
  final transformed = BoundaryResult(
    width: output.width,
    height: output.height,
    status: boundary.status,
    outerBoundary: [
      for (final p in boundary.outerBoundary)
        BoundaryPoint(
          position: move(p.position),
          visibility: p.visibility,
          confidence: p.confidence,
        ),
    ],
    boundarySegments: [
      for (final s in boundary.boundarySegments)
        BoundarySegment(
          move(s.start),
          move(s.end),
          control1: s.control1 == null ? null : move(s.control1!),
          control2: s.control2 == null ? null : move(s.control2!),
          isImageClip: s.isImageClip,
        ),
    ],
    corners: boundary.corners?.map(move).toList(),
    creaseLines: [
      for (final line in boundary.creaseLines) line.map(move).toList(),
    ],
    confidence: boundary.confidence,
    reasonCodes: boundary.reasonCodes,
    modelVersion: boundary.modelVersion,
    preprocessingVersion: boundary.preprocessingVersion,
  );
  return IsolatedDocument(
    Uint8List.fromList(img.encodePng(output)),
    transformed,
    FlattenGeometry(
      boundary.corners!,
      boundary.boundarySegments,
      source.width,
      source.height,
      followCurves: false,
    ).estimatedRatio,
    measurementPng: measurement == null
        ? null
        : Uint8List.fromList(img.encodePng(measurement)),
  );
});
