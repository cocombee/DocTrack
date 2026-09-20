import 'dart:io';
import 'dart:convert';

import 'package:tracker_engine/tracker_engine.dart';

void main(List<String> args) {
  final source = Directory(
    args.isNotEmpty
        ? args[0]
        : '../Scanner Integration/Predictions Physical V18',
  );
  final out = Directory(
    args.length > 1
        ? args[1]
        : '../Scanner Integration/Physical Review V18/Inset Geometry',
  )..createSync(recursive: true);
  TrackerPoint point(dynamic p) =>
      TrackerPoint((p['x'] as num).toDouble(), (p['y'] as num).toDouble());
  for (final f in source.listSync().whereType<File>().where(
    (f) => f.path.endsWith('.jpg.json'),
  )) {
    final d = jsonDecode(f.readAsStringSync());
    final segments = [
      for (final s in d['boundarySegments'])
        BoundarySegment(
          point(s['start']),
          point(s['end']),
          isImageClip: s['isImageClip'] == true,
          control1: s['control1'] == null ? null : point(s['control1']),
          control2: s['control2'] == null ? null : point(s['control2']),
        ),
    ];
    final inset = offsetBoundarySegments(segments, d['width'], d['height'], 4);
    File('${out.path}/${f.uri.pathSegments.last}').writeAsStringSync(
      jsonEncode({
        'width': d['width'],
        'height': d['height'],
        'segments': inset.map((s) => s.toJson()).toList(),
        'points': [
          for (final s in inset)
            for (var j = 0; j < (s.isCurve ? 64 : 1); j++)
              {
                'x': s.at(j / (s.isCurve ? 64 : 1)).x,
                'y': s.at(j / (s.isCurve ? 64 : 1)).y,
              },
        ],
      }),
    );
  }
}
