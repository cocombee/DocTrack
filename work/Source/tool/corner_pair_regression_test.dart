// ignore_for_file: avoid_print
import 'dart:io';
import 'dart:math' as math;
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  test(
    'weak mixed ID corner follows strong adjacent arc without rounding top',
    () {
      final manifest =
          jsonDecode(File('../Data/manifest.json').readAsStringSync()) as List;
      for (final e in manifest.where(
        (e) => e['name'] == 'midv500_003360.jpg',
      )) {
        final index = manifest.indexOf(e);
        final im = img.bakeOrientation(
          img.decodeImage(File('../Data/${e['image']}').readAsBytesSync())!,
        );
        final r = TrackerRaster(
          rgbBytes: im.getBytes(order: img.ChannelOrder.rgb),
          width: im.width,
          height: im.height,
          rowStride: im.width * 3,
          documentKind: DocumentKind.identityDocument,
        );
        final d = jsonDecode(
          File(
            '../Scanner Integration/Predictions Physical V18/${e['name']}.json',
          ).readAsStringSync(),
        );
        final q = (d['corners'] as List)
            .map((p) => TrackerPoint(p['x'], p['y']))
            .toList();
        print('IMAGE ${index + 1} reasons=${d['reasonCodes']}');
        final result = classifyPaperCorners(
          r,
          [
            for (var i = 0; i < 4; i++)
              for (var j = 0; j < 80; j++)
                BoundarySegment(q[i], q[(i + 1) % 4]).at(j / 80),
          ],
          q,
          rigid: true,
          matchAdjacentRounded: true,
          diagnostic: print,
        );
        final curves = result.where((s) => s.isCurve).toList();
        expect(curves.length, 2);
        double distance(TrackerPoint a, TrackerPoint b) => math.sqrt(
          math.pow((a.x - b.x) * im.width, 2) +
              math.pow((a.y - b.y) * im.height, 2),
        );
        final right = distance(curves[0].start, q[2]),
            left = distance(curves[1].start, q[3]);
        expect(left, closeTo(31, 0.01));
        expect(right / left, inInclusiveRange(.85, 1.15));
        expect(result.first.start.x, q[0].x);
        expect(result.first.start.y, q[0].y);
        expect(result.first.end.x, q[1].x);
        expect(result.first.end.y, q[1].y);
        final unconstrained = classifyPaperCorners(
          r,
          [
            for (var i = 0; i < 4; i++)
              for (var j = 0; j < 80; j++)
                BoundarySegment(q[i], q[(i + 1) % 4]).at(j / 80),
          ],
          q,
          rigid: false,
        );
        expect(
          unconstrained.where((s) => s.isCurve).length,
          lessThanOrEqualTo(2),
        );
        print('paired radii: right=$right left=$left');
      }
    },
  );
}
