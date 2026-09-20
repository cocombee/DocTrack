import 'dart:io';
import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  test('background crossing cannot bevel a sharp paper corner', () {
    final im = img.decodeImage(
      File(
        '../Data/originals/validation/filtered_document_segmentation_003188.jpg',
      ).readAsBytesSync(),
    )!;
    final d = jsonDecode(
      File(
        '../Scanner Integration/Predictions Physical V18/filtered_document_segmentation_003188.jpg.json',
      ).readAsStringSync(),
    );
    final q = (d['corners'] as List)
        .map((p) => TrackerPoint(p['x'], p['y']))
        .toList();
    final r = TrackerRaster(
      rgbBytes: im.getBytes(order: img.ChannelOrder.rgb),
      width: im.width,
      height: im.height,
      rowStride: im.width * 3,
    );
    final result = classifyPaperCorners(
      r,
      [
        for (var i = 0; i < 4; i++)
          for (var j = 0; j < 80; j++)
            BoundarySegment(q[i], q[(i + 1) % 4]).at(j / 80),
      ],
      q,
      rigid: true,
      sidePrimitives: [
        for (var i = 0; i < 4; i++) [BoundarySegment(q[i], q[(i + 1) % 4])],
      ],
    );
    expect(result.length, 4);
    expect(result[1].end.x, q[2].x);
    expect(result[1].end.y, q[2].y);
    expect(result[2].start.x, q[2].x);
    expect(result[2].start.y, q[2].y);
  });
}
