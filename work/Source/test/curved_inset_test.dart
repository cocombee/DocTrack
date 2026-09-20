import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';

void main() {
  test('bowed side retains a four source-pixel normal inset', () {
    const curve = BoundarySegment(
      TrackerPoint(.1, .2),
      TrackerPoint(.9, .2),
      control1: TrackerPoint(.3, .05),
      control2: TrackerPoint(.7, .05),
    );
    const path = [
      curve,
      BoundarySegment(TrackerPoint(.9, .2), TrackerPoint(.9, .9)),
      BoundarySegment(TrackerPoint(.9, .9), TrackerPoint(.1, .9)),
      BoundarySegment(TrackerPoint(.1, .9), TrackerPoint(.1, .2)),
    ];
    final inset = offsetBoundarySegments(path, 640, 640, 4);
    for (var j = 2; j < 19; j++) {
      final q = inset.first.at(j / 20),
          distance = List.generate(2001, (i) {
            final p = curve.at(i / 2000);
            return math.sqrt(
              math.pow((q.x - p.x) * 640, 2) + math.pow((q.y - p.y) * 640, 2),
            );
          }).reduce(math.min);
      expect(distance, closeTo(4, .35));
    }
    expect(inset.length, 4);
    expect(curve.start.x, .1);
  });
}
