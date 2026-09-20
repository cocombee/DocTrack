import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  for (final rounded in [
    <int>{0, 1, 2, 3},
    <int>{2, 3},
  ]) {
    test('extending a card preserves ${rounded.length} rounded corners', () {
      const corners = [
        TrackerPoint(.1, .2),
        TrackerPoint(.8, .2),
        TrackerPoint(.8, .8),
        TrackerPoint(.1, .8),
      ];
      const incoming = [
        TrackerPoint(.1, .25),
        TrackerPoint(.75, .2),
        TrackerPoint(.8, .75),
        TrackerPoint(.15, .8),
      ];
      const outgoing = [
        TrackerPoint(.15, .2),
        TrackerPoint(.8, .25),
        TrackerPoint(.75, .8),
        TrackerPoint(.1, .75),
      ];
      final segments = <BoundarySegment>[];
      for (var i = 0; i < 4; i++) {
        final next = (i + 1) % 4;
        if (rounded.contains(i)) {
          segments.add(
            BoundarySegment(
              incoming[i],
              outgoing[i],
              control1: corners[i],
              control2: corners[i],
            ),
          );
        }
        segments.add(
          BoundarySegment(
            rounded.contains(i) ? outgoing[i] : corners[i],
            rounded.contains(next) ? incoming[next] : corners[next],
          ),
        );
      }
      final extended = extendCardPrimitives(
        segments,
        corners,
        const [
          TrackerPoint(.1, .2),
          TrackerPoint(.9, .2),
          TrackerPoint(.9, .8),
          TrackerPoint(.1, .8),
        ],
        1000,
        1000,
      );
      expect(extended.length, segments.length);
      for (var i = 0; i < segments.length; i++) {
        final before = segments[i], after = extended[i];
        expect(after.isCurve, before.isCurve);
        if (!before.isCurve) continue;
        // A right corner translates by 100px; its radius, handles and shape
        // remain intact. Left corners do not move when the right side extends.
        final shift = before.start.x > .5 ? .1 : 0.0;
        for (var j = 0; j <= 20; j++) {
          expect(
            after.at(j / 20).x,
            closeTo(before.at(j / 20).x + shift, 1e-10),
          );
          expect(after.at(j / 20).y, closeTo(before.at(j / 20).y, 1e-10));
        }
      }
    });
  }
}
