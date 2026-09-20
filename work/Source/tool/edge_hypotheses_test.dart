import 'dart:io';
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  test('compare pixel edge search hypotheses', () {
    final manifest =
        jsonDecode(File('../Data/manifest.json').readAsStringSync()) as List;
    final e = manifest[47];
    final im = img.bakeOrientation(
      img.decodeImage(File('../Data/${e['image']}').readAsBytesSync())!,
    );
    final r = TrackerRaster(
      rgbBytes: im.getBytes(order: img.ChannelOrder.rgb),
      width: im.width,
      height: im.height,
      rowStride: im.width * 3,
    );
    final d = jsonDecode(
      File('../Scanner Integration/Predictions Simple V17/${e['name']}.json')
          .readAsStringSync(),
    );
    final q = (d['corners'] as List)
        .map((p) => TrackerPoint(p['x'], p['y']))
        .toList();
    final dir = Directory('../Scanner Integration/Edge Hypotheses')
      ..createSync(recursive: true);
    for (final radius in [40.0, 80.0, 120.0]) {
      for (final pin in [true, false]) {
        final raw = refineDocumentBoundary(
          r,
          q,
          roundCorners: false,
          searchRadius: radius,
          pinCorners: pin,
          exteriorEvidence: true,
        );
        final segments = fitBoundaryCurves(
          raw,
          r.width,
          r.height,
          tolerance: 2,
        );
        File('${dir.path}/$radius-$pin.json').writeAsStringSync(
          jsonEncode({
            'width': r.width,
            'height': r.height,
            'segments': segments.map((s) => s.toJson()).toList(),
          }),
        );
      }
    }
  });
}
