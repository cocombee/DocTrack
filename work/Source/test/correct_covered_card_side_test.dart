import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  test('side correction preserves accepted arc and emits one opposite sharp corner', () {
    const q = [
      TrackerPoint(.1, .1),
      TrackerPoint(.8, .1),
      TrackerPoint(.8, .8),
      TrackerPoint(.1, .8),
    ];
    const updated = [
      TrackerPoint(.1, .1),
      TrackerPoint(.9, .1),
      TrackerPoint(.9, .8),
      TrackerPoint(.1, .8),
    ];
    const arc = BoundarySegment(
      TrackerPoint(.1, .12),
      TrackerPoint(.12, .1),
      control1: TrackerPoint(.1, .105),
      control2: TrackerPoint(.105, .1),
    );
    const segments = [
      arc,
      BoundarySegment(TrackerPoint(.12, .1), TrackerPoint(.8, .1)),
      BoundarySegment(TrackerPoint(.8, .1), TrackerPoint(.8, .77)),
      BoundarySegment(TrackerPoint(.8, .77), TrackerPoint(.77, .8)),
      BoundarySegment(TrackerPoint(.77, .8), TrackerPoint(.1, .8)),
      BoundarySegment(TrackerPoint(.1, .8), TrackerPoint(.1, .12)),
    ];
    final result = correctCoveredCardSide(segments, q, updated, 800, 600);
    expect(result.length, 5);
    expect(result.where((s) => s.isCurve).length, 1);
    expect(result.first.control1!.x, arc.control1!.x);
    expect(result.first.control2!.y, arc.control2!.y);
    expect(result.where((s) => s.start.x == .9 && s.start.y == .8).length, 1);
    expect(result[2].end.x, .9);
    expect(result[2].end.y, .8);
    expect(result[3].isCurve, isFalse);
  });
}
