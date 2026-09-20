import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  List<TrackerPoint> project(double g, double h) => [
    for (final p in const [
      TrackerPoint(0, 0),
      TrackerPoint(1, 0),
      TrackerPoint(1, 1),
      TrackerPoint(0, 1),
    ])
      TrackerPoint(
        p.x / (1 + g * p.x + h * p.y),
        p.y / (1 + g * p.x + h * p.y),
      ),
  ];
  test('flat corners have equal apparent scales', () {
    expect(
      projectedCornerScales(project(0, 0)),
      everyElement(closeTo(1, 1e-12)),
    );
  });
  test('one-axis perspective produces near and far pairs', () {
    final s = projectedCornerScales(project(0, .3));
    expect(s[0], closeTo(s[1], 1e-12));
    expect(s[2], closeTo(s[3], 1e-12));
    expect(s[0], greaterThan(s[2]));
  });
  test('diagonal perspective produces largest, two intermediate, smallest', () {
    final s = projectedCornerScales(project(.2, .2));
    expect(s[0], greaterThan(s[1]));
    expect(s[1], closeTo(s[3], 1e-12));
    expect(s[1], greaterThan(s[2]));
  });
}
