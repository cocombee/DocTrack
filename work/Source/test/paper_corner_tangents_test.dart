import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  test('paper main corner uses the local folded-side tangent', () {
    const w = 600, h = 500;
    const polygon = [
      TrackerPoint(.22, .18),
      TrackerPoint(.84, .15),
      TrackerPoint(.82, .85),
      TrackerPoint(.15, .8),
      TrackerPoint(.16, .4),
    ];
    final bytes = Uint8List(w * h * 3);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        var inside = false;
        final px = x / w, py = y / h;
        for (var i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
          final a = polygon[i], b = polygon[j];
          if ((a.y > py) != (b.y > py) &&
              px < (b.x - a.x) * (py - a.y) / (b.y - a.y) + a.x) {
            inside = !inside;
          }
        }
        for (var c = 0; c < 3; c++) {
          bytes[(y * w + x) * 3 + c] = inside ? 230 : 35;
        }
      }
    }
    final corrected = refineSharpPaperCornerTangents(
      TrackerRaster(rgbBytes: bytes, width: w, height: h, rowStride: w * 3),
      [const TrackerPoint(.24, .195), ...polygon.skip(1).take(3)],
    );
    for (var i = 0; i < 4; i++) {
      final dx = (corrected[i].x - polygon[i].x) * w;
      final dy = (corrected[i].y - polygon[i].y) * h;
      expect(math.sqrt(dx * dx + dy * dy), lessThan(3));
    }
  });
  test('no corner evidence means no invented corner movement', () {
    final points = [
      const TrackerPoint(.2, .2),
      const TrackerPoint(.8, .2),
      const TrackerPoint(.8, .8),
      const TrackerPoint(.2, .8),
    ];
    final result = refineSharpPaperCornerTangents(
      TrackerRaster(
        rgbBytes: Uint8List(200 * 200 * 3),
        width: 200,
        height: 200,
        rowStride: 600,
      ),
      points,
    );
    for (var i = 0; i < 4; i++) {
      expect(result[i].x, points[i].x);
      expect(result[i].y, points[i].y);
    }
  });
}
