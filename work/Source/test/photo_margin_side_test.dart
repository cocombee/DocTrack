import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  const q = [
    TrackerPoint(.125, 1 / 6),
    TrackerPoint(.725, 1 / 6),
    TrackerPoint(.725, .7),
    TrackerPoint(.125, .7),
  ];
  TrackerRaster scene(bool gap) {
    final pixels = Uint8List(800 * 600 * 3);
    for (var y = 0; y < 600; y++) {
      for (var x = 0; x < 800; x++) {
        var rgb = <int>[200, 155, 130];
        if (x >= 100 && x < 596 && y >= 100 && y < 420) rgb = [165, 165, 165];
        if (x >= 430 && x < 580 && y >= 121 && y < 330) rgb = [220, 220, 220];
        if (x >= 596 && x < 605 && y >= 100 && y < 420) {
          rgb = gap ? [20, 20, 20] : [180, 180, 180];
        }
        for (var c = 0; c < 3; c++) {
          pixels[(y * 800 + x) * 3 + c] = rgb[c];
        }
      }
    }
    return TrackerRaster(
      rgbBytes: pixels,
      width: 800,
      height: 600,
      rowStride: 2400,
      documentKind: DocumentKind.identityDocument,
    );
  }

  test('photo margin caps the side before dark gap and hand', () {
    final result = refinePhotoMarginSide(scene(true), q);
    expect(result, isNotNull);
    final shift = (result![1].x - q[1].x) * 800;
    expect(shift, greaterThan(8));
    expect(shift, lessThanOrEqualTo(21.5));
    expect(result[0].x, q[0].x);
    expect(result[3].y, q[3].y);
  });
  test('photo margin alone cannot move a side without gap evidence', () {
    expect(refinePhotoMarginSide(scene(false), q), isNull);
  });
}
