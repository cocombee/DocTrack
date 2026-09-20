import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  test(
    'coloured panels remain inside paper despite stronger printed contrast',
    () {
      const w = 400, h = 300;
      final rgb = Uint8List(w * h * 3);
      for (var y = 0; y < h; y++) {
        for (var x = 0; x < w; x++) {
          final inside = x >= 40 && x <= 360 && y >= 40 && y <= 260;
          final color = !inside
              ? [160, 155, 145]
              : y > 230
              ? [30, 70, 190]
              : [235, 235, 230];
          for (var c = 0; c < 3; c++) {
            rgb[(y * w + x) * 3 + c] = color[c];
          }
        }
      }
      final shape = refineVerifiedPaper(
        TrackerRaster(rgbBytes: rgb, width: w, height: h, rowStride: w * 3),
        const [
          TrackerPoint(.1, 40 / h),
          TrackerPoint(.9, 40 / h),
          TrackerPoint(.9, 260 / h),
          TrackerPoint(.1, 260 / h),
        ],
      );
      final bottom = [
        for (final s in shape.segments)
          for (var j = 0; j < 20; j++) s.at(j / 20),
      ].where((p) => p.x > .2 && p.x < .8 && p.y > .6);
      expect(bottom, isNotEmpty);
      expect(bottom.every((p) => (p.y * h - 260).abs() < 3), isTrue);
    },
  );
  test('square coloured paper retains sharp geometry', () {
    const w = 400, h = 300;
    final rgb = Uint8List(w * h * 3);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final inside = x >= 40 && x <= 360 && y >= 40 && y <= 260;
        final color = inside ? [220, 130, 160] : [40, 40, 40];
        for (var c = 0; c < 3; c++) {
          rgb[(y * w + x) * 3 + c] = color[c];
        }
      }
    }
    final shape = refineVerifiedPaper(
      TrackerRaster(rgbBytes: rgb, width: w, height: h, rowStride: w * 3),
      const [
        TrackerPoint(.1, 40 / h),
        TrackerPoint(.9, 40 / h),
        TrackerPoint(.9, 260 / h),
        TrackerPoint(.1, 260 / h),
      ],
    );
    expect(shape.segments.length, 4);
    expect(shape.segments.every((s) => !s.isCurve), isTrue);
  });
}
