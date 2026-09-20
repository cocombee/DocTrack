import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';

void main() {
  test('open side fit preserves endpoints without a false closing edge', () {
    final points = [
      for (var i = 0; i <= 30; i++)
        TrackerPoint(.1 + i * .02, .3 + .05 * (i / 30) * (1 - i / 30)),
    ];
    final result = fitBoundaryCurves(
      points,
      640,
      640,
      closed: false,
      preserveStructuralTurns: true,
    );
    expect(result.first.start.x, points.first.x);
    expect(result.last.end.x, points.last.x);
    for (var i = 0; i < result.length; i++) {
      expect(result[i].end.x, greaterThan(result[i].start.x));
      if (i > 0) {
        expect(result[i].start.x, result[i - 1].end.x);
        expect(result[i].start.y, result[i - 1].end.y);
      }
    }
  });

  test('engine exports sparse cubic geometry and preserves square corners', () {
    final square = [
      const TrackerPoint(.1, .1),
      const TrackerPoint(.9, .1),
      const TrackerPoint(.9, .9),
      const TrackerPoint(.1, .9),
    ];
    final lines = fitBoundaryCurves(square, 1000, 1000);
    expect(lines.length, 4);
    expect(lines.every((s) => !s.isCurve), isTrue);
    final circle = List.generate(180, (i) {
      final a = 2 * math.pi * i / 180;
      return TrackerPoint(.5 + .3 * math.cos(a), .5 + .3 * math.sin(a));
    });
    final result = BoundaryResult(
      width: 1000,
      height: 1000,
      status: BoundaryStatus.uncertain,
      outerBoundary: circle
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
    expect(result.boundarySegments.length, lessThan(20));
    expect(result.boundarySegments.any((s) => s.isCurve), isTrue);
    for (final segment in result.boundarySegments) {
      for (var i = 0; i <= 30; i++) {
        final p = segment.at(i / 30);
        final radius =
            math.sqrt(math.pow(p.x - .5, 2) + math.pow(p.y - .5, 2)) * 1000;
        expect((radius - 300).abs(), lessThan(2));
      }
    }
    expect(result.toJson()['boundarySegments'], isNotEmpty);
  });
  test('engine offset preserves cubics and four-pixel square inset', () {
    const points = [
      TrackerPoint(.1, .1),
      TrackerPoint(.9, .1),
      TrackerPoint(.9, .9),
      TrackerPoint(.1, .9),
    ];
    final base = fitBoundaryCurves(points, 1000, 500);
    final inset = offsetBoundarySegments(base, 1000, 500, 4);
    for (var i = 0; i < base.length; i++) {
      expect((inset[i].start.x - base[i].start.x).abs(), closeTo(.004, 1e-9));
      expect((inset[i].start.y - base[i].start.y).abs(), closeTo(.008, 1e-9));
    }
    final circle = List.generate(180, (i) {
      final a = 2 * math.pi * i / 180;
      return TrackerPoint(.5 + .3 * math.cos(a), .5 + .3 * math.sin(a));
    });
    final curves = fitBoundaryCurves(circle, 1000, 1000),
        offset = offsetBoundarySegments(curves, 1000, 1000, 4);
    expect(offset.length, curves.length);
    for (var k = 0; k < offset.length; k++) {
      expect(offset[k].isCurve, curves[k].isCurve);
      for (var i = 0; i <= 20; i++) {
        final p = offset[k].at(i / 20);
        final radius =
            math.sqrt(math.pow(p.x - .5, 2) + math.pow(p.y - .5, 2)) * 1000;
        expect((radius - 296).abs(), lessThan(2));
      }
    }
  });
}
