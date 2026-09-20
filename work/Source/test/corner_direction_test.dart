import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';

void main() {
  test('corner cannot reverse from incoming side or exceed its tangent intersection', () {
    const path = [
      BoundarySegment(TrackerPoint(.1, .1), TrackerPoint(.1, .7)),
      BoundarySegment(
        TrackerPoint(.1, .7),
        TrackerPoint(.2, .8),
        control1: TrackerPoint(.06, .76),
        control2: TrackerPoint(.0, .8),
      ),
      BoundarySegment(TrackerPoint(.2, .8), TrackerPoint(.9, .8)),
      BoundarySegment(TrackerPoint(.9, .8), TrackerPoint(.9, .1)),
      BoundarySegment(TrackerPoint(.9, .1), TrackerPoint(.1, .1)),
    ];
    final result = constrainCornerDirections(path, 1000, 1000), arc = result[1];
    expect(arc.control1!.x, closeTo(.1, 1e-12));
    expect(arc.control2!.x, greaterThanOrEqualTo(.1));
    for (var j = 0; j <= 100; j++) {
      final p = arc.at(j / 100);
      expect(p.x, greaterThanOrEqualTo(.1 - 1e-12));
      expect(p.y, lessThanOrEqualTo(.8 + 1e-12));
    }
    expect(identical(result.first, path.first), isTrue);
  });
  test('sharp top-right inset stays within the intended corner', () {
    const path = [
      BoundarySegment(TrackerPoint(.1, .1), TrackerPoint(.9, .1)),
      BoundarySegment(TrackerPoint(.9, .1), TrackerPoint(.9, .8)),
      BoundarySegment(TrackerPoint(.9, .8), TrackerPoint(.1, .9)),
      BoundarySegment(TrackerPoint(.1, .9), TrackerPoint(.1, .1)),
    ];
    final inset = offsetBoundarySegments(path, 100, 100, 4);
    expect(inset.first.end.x, closeTo(.86, 1e-12));
    expect(inset.first.end.y, closeTo(.14, 1e-12));
  });
  test('legitimate bowed side is not treated as a corner', () {
    const shape = [
      BoundarySegment(
        TrackerPoint(.1, .2),
        TrackerPoint(.9, .2),
        control1: TrackerPoint(.3, .05),
        control2: TrackerPoint(.7, .05),
      ),
      BoundarySegment(TrackerPoint(.9, .2), TrackerPoint(.9, .8)),
      BoundarySegment(TrackerPoint(.9, .8), TrackerPoint(.1, .8)),
      BoundarySegment(TrackerPoint(.1, .8), TrackerPoint(.1, .2)),
    ];
    expect(
      identical(
        constrainCornerDirections(shape, 1000, 1000).first,
        shape.first,
      ),
      isTrue,
    );
  });
}
