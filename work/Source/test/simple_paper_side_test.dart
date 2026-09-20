import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  const a = TrackerPoint(.2, .1), b = TrackerPoint(.2, .9);
  test('supported straight paper side needs no extra anchor', () {
    final side = fitSimplePaperSide(
      a,
      b,
      500,
      500,
      30,
      (t, d) => 35 * math.exp(-d * d / 8),
    );
    expect(side, hasLength(1));
    expect(side!.single.isCurve, isFalse);
  });
  test('paper fold keeps a sharp junction between two curved sections', () {
    double expected(double t) {
      final u = t < .5 ? t * 2 : (t - .5) * 2;
      return 18 * (t < .5 ? u : 1 - u) - 8 * 4 * u * (1 - u);
    }

    final side = fitSimplePaperSide(
      a,
      b,
      500,
      500,
      30,
      (t, d) => 35 * math.exp(-math.pow(d - expected(t), 2) / 8),
    );
    expect(side, hasLength(2));
    expect(side!.every((s) => s.isCurve), isTrue);
    expect(side.first.end.y, closeTo(.5, .025));
    expect(side.first.end.x, closeTo(side.last.start.x, 1e-9));
    for (final s in side) {
      for (var j = 0; j <= 40; j++) {
        final p = s.at(j / 40), t = (p.y - .1) / .8;
        expect((.2 - p.x) * 500, closeTo(expected(t), 2));
        expect(((p.x - .2) * 500).abs(), lessThanOrEqualTo(30));
      }
    }
  });
  test('unsupported side cannot be simplified merely to reduce points', () {
    expect(
      fitSimplePaperSide(
        a,
        b,
        500,
        500,
        30,
        (t, d) => t > .3 && t < .65 ? 0 : 35 * math.exp(-d * d / 8),
      ),
      isNull,
    );
  });
}
