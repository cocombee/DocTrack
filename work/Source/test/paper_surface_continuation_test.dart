import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

TrackerRaster surface({bool fold = false, bool sameStock = false}) {
  const w = 400, h = 400;
  final bytes = Uint8List(w * h * 3);
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      final left = fold && y > 170 && y < 240 ? 100 : 80;
      final paper = x >= left && x <= 320 && y >= 60 && y <= 340;
      final underlying = x >= 55 && x < left && y > 120 && y < 300;
      final color = paper || underlying && sameStock
          ? [210, 156, 181]
          : underlying
          ? [235, 235, 235]
          : [75, 52, 38];
      for (var c = 0; c < 3; c++) {
        bytes[(y * w + x) * 3 + c] = color[c];
      }
    }
  }
  return TrackerRaster(rgbBytes: bytes, width: w, height: h, rowStride: w * 3);
}

void main() {
  const bottom = TrackerPoint(.2, .85), top = TrackerPoint(.2, .15);
  test(
    'straight coloured paper remains separate from an underlying white sheet',
    () {
      expect(supportsStraightPaperSurface(surface(), bottom, top), isTrue);
    },
  );
  test('real boundary fold prevents the straight-surface fallback', () {
    expect(
      supportsStraightPaperSurface(surface(fold: true), bottom, top),
      isFalse,
    );
  });
  test(
    'ambiguous same-colour material does not justify a forced straight edge',
    () {
      expect(
        supportsStraightPaperSurface(surface(sameStock: true), bottom, top),
        isFalse,
      );
    },
  );
}
