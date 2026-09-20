import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

TrackerRaster raster([List<TrackerPoint>? crop]) => TrackerRaster(
  rgbBytes: Uint8List(30000),
  width: 100,
  height: 100,
  rowStride: 300,
  validImagePolygon: crop,
);
void main() {
  test('fully visible boundary is unchanged even when a handle is outside', () {
    const a = TrackerPoint(.2, .3),
        b = TrackerPoint(.2, .7),
        c = TrackerPoint(.8, .5);
    final shape = [
      BoundarySegment(
        a,
        b,
        control1: const TrackerPoint(-.01, .4),
        control2: const TrackerPoint(.2, .6),
      ),
      BoundarySegment(b, c),
      BoundarySegment(c, a),
    ];
    final result = constrainToImageCrop(raster(), shape);
    expect(result.length, 3);
    for (var i = 0; i < 3; i++) {
      expect(identical(result[i], shape[i]), isTrue);
    }
  });
  test('partial curve is trimmed, preserving the original visible curve', () {
    const a = TrackerPoint(-.2, .2),
        b = TrackerPoint(.6, .6),
        c = TrackerPoint(.6, .8),
        d = TrackerPoint(-.2, .8);
    final curve = BoundarySegment(
      a,
      b,
      control1: const TrackerPoint(.1, .2),
      control2: const TrackerPoint(.6, .3),
    );
    final result = constrainToImageCrop(raster(), [
      curve,
      BoundarySegment(b, c),
      BoundarySegment(c, d),
      BoundarySegment(d, a),
    ]);
    final arc = result.singleWhere((s) => s.isCurve);
    expect(arc.start.x, closeTo(0, 1e-10));
    expect(arc.end.x, b.x);
    var lo = 0.0, hi = 1.0;
    for (var i = 0; i < 60; i++) {
      final t = (lo + hi) / 2;
      if (curve.at(t).x < 0) {
        lo = t;
      } else {
        hi = t;
      }
    }
    for (var j = 0; j <= 20; j++) {
      final p = arc.at(j / 20), q = curve.at(hi + (1 - hi) * j / 20);
      expect(p.x, closeTo(q.x, 1e-9));
      expect(p.y, closeTo(q.y, 1e-9));
    }
    expect(result.where((s) => s.isImageClip).length, 1);
    for (var i = 0; i < result.length; i++) {
      expect(
        result[i].end.x,
        closeTo(result[(i + 1) % result.length].start.x, 1e-9),
      );
      expect(
        result[i].end.y,
        closeTo(result[(i + 1) % result.length].start.y, 1e-9),
      );
    }
  });
  test('original rotated crop cuts straight edges at exact intersections', () {
    const frame = [
      TrackerPoint(0, 0),
      TrackerPoint(1, 0),
      TrackerPoint(1, 1),
      TrackerPoint(0, 1),
    ];
    const crop = [
      TrackerPoint(.5, .1),
      TrackerPoint(.9, .5),
      TrackerPoint(.5, .9),
      TrackerPoint(.1, .5),
    ];
    final result = constrainToImageCrop(raster(crop), [
      for (var i = 0; i < 4; i++) BoundarySegment(frame[i], frame[(i + 1) % 4]),
    ]);
    expect(result.length, 4);
    expect(result.every((s) => s.isImageClip), isTrue);
    for (final p in crop) {
      expect(
        result.any(
          (s) => (s.start.x - p.x).abs() + (s.start.y - p.y).abs() < 1e-8,
        ),
        isTrue,
      );
    }
  });
  test('curve exits and re-enters with both endpoints inside', () {
    const a = TrackerPoint(.1, .2),
        b = TrackerPoint(.1, .8),
        c = TrackerPoint(.8, .5);
    final shape = [
      BoundarySegment(
        a,
        b,
        control1: const TrackerPoint(-.4, .3),
        control2: const TrackerPoint(-.4, .7),
      ),
      BoundarySegment(b, c),
      BoundarySegment(c, a),
    ];
    final result = constrainToImageCrop(raster(), shape);
    expect(result.where((s) => s.isCurve).length, 2);
    expect(result.where((s) => s.isImageClip).length, 1);
    for (final s in result) {
      for (var j = 0; j <= 20; j++) {
        expect(s.at(j / 20).x, greaterThanOrEqualTo(-1e-9));
      }
    }
  });
}
