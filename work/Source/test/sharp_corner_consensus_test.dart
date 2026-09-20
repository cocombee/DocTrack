import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  const q = [
    TrackerPoint(.125, .125),
    TrackerPoint(.875, .125),
    TrackerPoint(.875, .875),
    TrackerPoint(.125, .875),
  ];
  TrackerRaster scene(Set<int> arcs, bool faint, [Set<int> hidden = const {}]) {
    final pixels = Uint8List(400 * 400 * 3);
    for (var y = 0; y < 400; y++) {
      for (var x = 0; x < 400; x++) {
        var inside = x >= 50 && x <= 350 && y >= 50 && y <= 350;
        var weak = false;
        for (final i in arcs) {
          final dx = i == 0 || i == 3 ? x - 50 : 350 - x;
          final dy = i < 2 ? y - 50 : 350 - y;
          if (dx < 35 && dy < 35) {
            weak = faint;
            if (dx >= 0 &&
                dy >= 0 &&
                dx < 20 &&
                dy < 20 &&
                math.sqrt(dx / 20) + math.sqrt(dy / 20) < 1) {
              inside = false;
            }
          }
        }
        var value = inside ? 210 : (weak ? 190 : 20);
        for (final i in hidden) {
          final cx = i == 0 || i == 3 ? 50 : 350;
          final cy = i < 2 ? 50 : 350;
          if ((x - cx).abs() < 40 && (y - cy).abs() < 40) value = 130;
        }
        for (var c = 0; c < 3; c++) {
          pixels[(y * 400 + x) * 3 + c] = value;
        }
      }
    }
    return TrackerRaster(
      rgbBytes: pixels,
      width: 400,
      height: 400,
      rowStride: 1200,
      documentKind: DocumentKind.identityDocument,
    );
  }

  for (final arcs in [
    <int>{2},
    <int>{2, 3},
    <int>{1, 3},
  ]) {
    test('clear sharp corners reject uncertain arcs $arcs', () {
      final messages = <String>[];
      final segments = classifyPaperCorners(
        scene(arcs, true),
        q,
        q,
        rigid: true,
        allowCuts: false,
        diagnostic: messages.add,
      );
      expect(segments.where((s) => s.control1 != null), isEmpty);
      expect(
        messages.any((s) => s.contains('CLEAR_SHARP_CORNER_CONSENSUS')),
        isTrue,
      );
    });
  }
  test('visible exceptional rounding survives sharp corner consensus', () {
    final segments = classifyPaperCorners(
      scene({2, 3}, false),
      q,
      q,
      rigid: true,
      allowCuts: false,
    );
    expect(segments.where((s) => s.control1 != null).length, 2);
  });
  for (final hidden in [
    <int>{2, 3},
    <int>{2},
  ]) {
    test('visible rounded corners predict covered corners $hidden', () {
      final messages = <String>[];
      final segments = classifyPaperCorners(
        scene({0, 1, 2, 3}, false, hidden),
        q,
        q,
        rigid: true,
        allowCuts: false,
        diagnostic: messages.add,
      );
      expect(segments.where((s) => s.control1 != null).length, 4);
      expect(
        messages.any((s) => s.contains('CLEAR_ROUNDED_CORNER_CONSENSUS')),
        isTrue,
      );
    });
  }
}
