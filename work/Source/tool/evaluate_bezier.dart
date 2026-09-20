import 'dart:io';
import 'dart:convert';

import 'package:tracker_engine/tracker_engine.dart';

void main() {
  final out = Directory('../Scanner Integration/Bezier Evaluation')
    ..createSync(recursive: true);
  for (final name in [
    'document_segmentation_tzzh7_001989.jpg',
    'filtered_document_segmentation_000995.jpg',
    'id_detections_004020.jpg',
  ]) {
    final d = jsonDecode(
      File('../Scanner Integration/Predictions Dart ONNX V10/$name.json')
          .readAsStringSync(),
    );
    final points = (d['outerBoundary'] as List)
        .map(
          (p) => TrackerPoint(
            (p['x'] as num).toDouble(),
            (p['y'] as num).toDouble(),
          ),
        )
        .toList();
    final segments = fitBoundaryCurves(points, d['width'], d['height']);
    File('${out.path}/$name.json').writeAsStringSync(
      jsonEncode({
        'width': d['width'],
        'height': d['height'],
        'segments': segments.map((s) => s.toJson()).toList(),
        'displaySegments': offsetBoundarySegments(
          segments,
          d['width'],
          d['height'],
          4,
        ).map((s) => s.toJson()).toList(),
      }),
    );
    stdout.writeln(
      '$name: ${points.length} samples -> ${segments.length} anchors, ${segments.where((s) => s.isCurve).length} curves',
    );
  }
}
