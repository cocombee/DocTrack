import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:pen_tool/flattening/flatten_geometry.dart';
import 'package:pen_tool/flattening/rounded_outline.dart';

void main() {
  const corners = [
    TrackerPoint(0, 0),
    TrackerPoint(1, 0),
    TrackerPoint(1, 1),
    TrackerPoint(0, 1),
  ];
  BoundarySegment arc(int corner, double radius) {
    final right = corner % 2 == 1, bottom = corner >= 2;
    TrackerPoint p(double x, double y) =>
        TrackerPoint(right ? 1 - x : x, bottom ? 1 - y : y);
    return BoundarySegment(
      p(0, radius),
      p(radius, 0),
      control1: p(0, radius * .45),
      control2: p(radius * .45, 0),
    );
  }

  test('two rounded corners share one output radius; other two stay sharp', () {
    final g = FlattenGeometry(
      corners,
      [arc(0, .06), arc(1, .09)],
      200,
      100,
      followCurves: false,
    );
    final shape = RoundedOutline(g, 200, 100);
    for (var x = 0; x < 20; x++) {
      for (var y = 0; y < 20; y++) {
        expect(
          shape.contains(x / 200, y / 100),
          shape.contains(1 - x / 200, y / 100),
        );
      }
    }
    expect(shape.contains(0, 1), isTrue);
    expect(shape.contains(1, 1), isTrue);
    expect(shape.contains(0, 0), isFalse);
  });
  test('four rounded corners match in output pixel space', () {
    final g = FlattenGeometry(
      corners,
      [for (var i = 0; i < 4; i++) arc(i, .05 + i * .01)],
      200,
      100,
      followCurves: false,
    );
    final shape = RoundedOutline(g, 200, 100);
    for (var x = 0; x < 20; x++) {
      for (var y = 0; y < 20; y++) {
        final expected = shape.contains(x / 200, y / 100);
        expect(shape.contains(1 - x / 200, y / 100), expected);
        expect(shape.contains(x / 200, 1 - y / 100), expected);
        expect(shape.contains(1 - x / 200, 1 - y / 100), expected);
      }
    }
  });
}
