import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:pen_tool/scanner/boundary_display.dart';

void main() {
  test(
    'inset uses source pixels for either winding without modifying engine data',
    () {
      final square = [
        const TrackerPoint(.1, .1),
        const TrackerPoint(.9, .1),
        const TrackerPoint(.9, .9),
        const TrackerPoint(.1, .9),
      ];
      for (final points in [square, square.reversed.toList()]) {
        final result = BoundaryResult(
          width: 1000,
          height: 500,
          status: BoundaryStatus.uncertain,
          outerBoundary: points
              .map(
                (p) => BoundaryPoint(
                  position: p,
                  visibility: BoundaryVisibility.inferred,
                  confidence: .5,
                ),
              )
              .toList(),
          confidence: .5,
          modelVersion: 'test',
          preprocessingVersion: 'test',
        );
        final inset = insetBoundary(result);
        for (var i = 0; i < points.length; i++) {
          expect((inset[i].x - points[i].x).abs(), closeTo(.004, 1e-9));
          expect((inset[i].y - points[i].y).abs(), closeTo(.008, 1e-9));
          expect(result.outerBoundary[i].position, same(points[i]));
        }
      }
    },
  );
}
