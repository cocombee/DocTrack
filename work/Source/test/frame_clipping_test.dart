import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';

void main() {
  test('camera closure stays on frame while physical edges inset', () {
    const p = [
      TrackerPoint(0, .1),
      TrackerPoint(.9, .1),
      TrackerPoint(.9, .9),
      TrackerPoint(0, .9),
    ];
    final result = offsetBoundarySegments(
      [for (var i = 0; i < 4; i++) BoundarySegment(p[i], p[(i + 1) % 4])],
      100,
      100,
      4,
    );
    expect(result[0].start.x, 0);
    expect(result[0].start.y, closeTo(.14, 1e-9));
    expect(result[2].end.x, 0);
    expect(result[2].end.y, closeTo(.86, 1e-9));
    expect(result[1].start.x, closeTo(.86, 1e-9));
  });
  test('two adjoining frame closures retain their image corner', () {
    const p = [
      TrackerPoint(0, 0),
      TrackerPoint(.9, 0),
      TrackerPoint(.9, .9),
      TrackerPoint(0, .9),
    ];
    final result = offsetBoundarySegments(
      [for (var i = 0; i < 4; i++) BoundarySegment(p[i], p[(i + 1) % 4])],
      100,
      100,
      4,
    );
    expect(result.first.start.x, 0);
    expect(result.first.start.y, 0);
  });
}
