import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  const corners = [
    TrackerPoint(.1, .15),
    TrackerPoint(.9, .15),
    TrackerPoint(.9, .85),
    TrackerPoint(.1, .85),
  ];
  final samples = [
    for (var i = 0; i < 4; i++)
      for (var j = 0; j < 80; j++)
        TrackerPoint(
          corners[i].x + (corners[(i + 1) % 4].x - corners[i].x) * j / 80,
          corners[i].y + (corners[(i + 1) % 4].y - corners[i].y) * j / 80,
        ),
  ];
  TrackerRaster raster({bool cut = false}) {
    final bytes = Uint8List(400 * 400 * 3);
    for (var y = 0; y < 400; y++) {
      for (var x = 0; x < 400; x++) {
        final inside =
            x >= 40 &&
            x <= 360 &&
            y >= 60 &&
            y <= 340 &&
            (!cut || x + y >= 116);
        for (var c = 0; c < 3; c++) {
          bytes[(y * 400 + x) * 3 + c] = inside ? 220 : 25;
        }
      }
    }
    return TrackerRaster(
      rgbBytes: bytes,
      width: 400,
      height: 400,
      rowStride: 1200,
    );
  }

  test('whole-side model keeps straight edges despite local dark printing', () {
    final r = raster();
    for (var y = 120; y < 160; y++) {
      for (var x = 348; x <= 360; x++) {
        for (var c = 0; c < 3; c++) {
          r.rgbBytes[(y * 400 + x) * 3 + c] = 0;
        }
      }
    }
    final modeled = fitPrimitiveSides(r, corners);
    expect(modeled.length, 320);
    for (var j = 0; j < 80; j++) {
      expect(modeled[80 + j].x, closeTo(.9, 1e-9));
    }
  });
  test('whole-side model retains a sustained paper bow', () {
    final bytes = Uint8List(400 * 400 * 3);
    for (var y = 0; y < 400; y++) {
      for (var x = 0; x < 400; x++) {
        final t = (y - 60) / 280;
        final right = 360 - 24 * 4 * t * (1 - t);
        final inside = x >= 40 && x <= right && y >= 60 && y <= 340;
        for (var c = 0; c < 3; c++) {
          bytes[(y * 400 + x) * 3 + c] = inside ? 220 : 25;
        }
      }
    }
    final r = TrackerRaster(
      rgbBytes: bytes,
      width: 400,
      height: 400,
      rowStride: 1200,
    );
    final primitives = <List<BoundarySegment>>[];
    final modeled = fitPrimitiveSides(r, corners, sidePrimitives: primitives);
    expect(modeled[120].x, closeTo(.84, .008));
    expect(modeled[80].x, .9);
    // Final primitives must represent the selected model itself, not a second
    // generic curve approximation of its samples.
    for (var j = 0; j < 80; j++) {
      final point = modeled[80 + j];
      final segment = primitives[1].firstWhere(
        (s) => point.y >= s.start.y - 1e-10 && point.y <= s.end.y + 1e-10,
      );
      final t = (point.y - segment.start.y) / (segment.end.y - segment.start.y);
      expect(segment.at(t).x, closeTo(point.x, 1e-9));
      expect(segment.at(t).y, closeTo(point.y, 1e-9));
    }
  });

  test('rigid plain rectangle has exactly four main straight sides', () {
    final segments = classifyPaperCorners(
      raster(),
      samples,
      corners,
      rigid: true,
    );
    expect(segments.length, 4);
    expect(segments.every((s) => !s.isCurve), isTrue);
  });
  test(
    'visible chamfer adds a short fifth side without unrestricted contour',
    () {
      final segments = classifyPaperCorners(
        raster(cut: true),
        samples,
        corners,
        rigid: true,
      );
      expect(segments.length, 5);
      expect(segments.every((s) => !s.isCurve), isTrue);
    },
  );
  test('occlusion mode cannot add a chamfer exception', () {
    final segments = classifyPaperCorners(
      raster(cut: true),
      samples,
      corners,
      rigid: true,
      allowCuts: false,
    );
    expect(segments.where((s) => !s.isCurve).length, 4);
    expect(segments.length, lessThanOrEqualTo(8));
  });

  TrackerRaster padded(
    int shade, {
    bool interrupted = false,
    bool explicitCrop = false,
  }) {
    final bytes = Uint8List(400 * 400 * 3)..fillRange(0, 400 * 400 * 3, 180);
    for (var y = 0; y < 400; y++) {
      for (var x = 328; x < 400; x++) {
        for (var c = 0; c < 3; c++) {
          bytes[(y * 400 + x) * 3 + c] = shade;
        }
      }
    }
    if (interrupted) {
      for (var y = 0; y < 400; y++) {
        for (var c = 0; c < 3; c++) {
          bytes[(y * 400 + 380) * 3 + c] = 100;
        }
      }
    }
    return TrackerRaster(
      rgbBytes: bytes,
      width: 400,
      height: 400,
      rowStride: 1200,
      validImagePolygon: explicitCrop
          ? const [
              TrackerPoint(0, 0),
              TrackerPoint(.82, 0),
              TrackerPoint(.82, 1),
              TrackerPoint(0, 1),
            ]
          : null,
    );
  }

  final cropCandidate = [
    const BoundarySegment(TrackerPoint(.1, .15), TrackerPoint(.83, .15)),
    const BoundarySegment(TrackerPoint(.83, .15), TrackerPoint(.83, .85)),
    const BoundarySegment(TrackerPoint(.83, .85), TrackerPoint(.1, .85)),
    const BoundarySegment(TrackerPoint(.1, .85), TrackerPoint(.1, .15)),
  ];
  test(
    'uninterrupted explicit crop metadata snaps and receives zero inset',
    () {
      final snapped = constrainToImageCrop(
        padded(0, explicitCrop: true),
        cropCandidate,
      );
      expect(snapped[1].isImageClip, isTrue);
      expect(snapped[1].start.x, closeTo(.82, 1e-9));
      final inset = offsetBoundarySegments(snapped, 400, 400, 4);
      expect(inset[1].start.x, closeTo(.82, 1e-9));
      expect(inset[1].end.x, closeTo(.82, 1e-9));
    },
  );
  test(
    'black, near-black and interrupted pixels alone are not crop padding',
    () {
      for (final r in [padded(0), padded(1), padded(0, interrupted: true)]) {
        final shape = constrainToImageCrop(r, cropCandidate);
        expect(shape.any((s) => s.isImageClip), isFalse);
        expect(shape[1].start.x, .83);
      }
    },
  );

  test('nearby material does not snap a completed shape to the frame', () {
    const q = [
      TrackerPoint(.2, .02),
      TrackerPoint(.8, .02),
      TrackerPoint(.8, .8),
      TrackerPoint(.2, .8),
    ];
    final continuous = Uint8List(400 * 400 * 3)
      ..fillRange(0, 400 * 400 * 3, 180);
    final r = TrackerRaster(
      rgbBytes: continuous,
      width: 400,
      height: 400,
      rowStride: 1200,
    );
    expect(
      constrainToImageCrop(r, [
            for (var i = 0; i < 4; i++) BoundarySegment(q[i], q[(i + 1) % 4]),
          ]).first.start.y ==
          .02,
      isTrue,
    );
    for (var y = 0; y < 5; y++) {
      for (var x = 0; x < 400; x++) {
        for (var c = 0; c < 3; c++) {
          continuous[(y * 400 + x) * 3 + c] = 50;
        }
      }
    }
    expect(
      constrainToImageCrop(r, [
        for (var i = 0; i < 4; i++) BoundarySegment(q[i], q[(i + 1) % 4]),
      ]).first.start.y,
      .02,
    );
  });

  test('rounded corners remain arcs rather than cut exceptions', () {
    final bytes = Uint8List(400 * 400 * 3);
    for (var y = 0; y < 400; y++) {
      for (var x = 0; x < 400; x++) {
        final dx = x - x.clamp(60, 340), dy = y - y.clamp(80, 320);
        final inside = dx * dx + dy * dy <= 400;
        for (var c = 0; c < 3; c++) {
          bytes[(y * 400 + x) * 3 + c] = inside ? 220 : 25;
        }
      }
    }
    final r = TrackerRaster(
      rgbBytes: bytes,
      width: 400,
      height: 400,
      rowStride: 1200,
    );
    final shape = classifyPaperCorners(r, samples, corners, rigid: true);
    expect(shape.length, 8);
    expect(shape.where((s) => s.isCurve).length, 4);
  });

  test(
    'predicted geometry outside the crop is constrained to its hard limit',
    () {
      final outside = [
        const BoundarySegment(TrackerPoint(.1, .15), TrackerPoint(.95, .15)),
        const BoundarySegment(TrackerPoint(.95, .15), TrackerPoint(.95, .85)),
        const BoundarySegment(TrackerPoint(.95, .85), TrackerPoint(.1, .85)),
        const BoundarySegment(TrackerPoint(.1, .85), TrackerPoint(.1, .15)),
      ];
      final shape = constrainToImageCrop(
        padded(0, explicitCrop: true),
        outside,
      );
      expect(shape[1].isImageClip, isTrue);
      for (final segment in shape) {
        for (var j = 0; j <= 32; j++) {
          {
            expect(segment.at(j / 32).x, lessThanOrEqualTo(.82 + 1e-9));
          }
        }
      }
    },
  );

  test(
    'a full-width dark printed band remains inside paper near the frame',
    () {
      final bytes = Uint8List(400 * 400 * 3);
      for (var y = 0; y < 400; y++) {
        for (var x = 0; x < 400; x++) {
          final inside = x >= 60 && x <= 340 && y >= 80 && y < 394;
          final value = inside
              ? (y >= 350 ? 35 : 220)
              : (((x ~/ 2 + y ~/ 2) % 2 == 0) ? 120 : 160);
          for (var c = 0; c < 3; c++) {
            bytes[(y * 400 + x) * 3 + c] = value;
          }
        }
      }
      final r = TrackerRaster(
        rgbBytes: bytes,
        width: 400,
        height: 400,
        rowStride: 1200,
      );
      const q = [
        TrackerPoint(.15, .2),
        TrackerPoint(.85, .2),
        TrackerPoint(.85, .9),
        TrackerPoint(.15, .9),
      ];
      final points = refineDocumentBoundary(
        r,
        q,
        roundCorners: false,
        pinCorners: false,
        searchRadius: 60,
      );
      final bottom =
          points
              .where((p) => p.x > .3 && p.x < .7 && p.y > .7)
              .map((p) => p.y * 400)
              .toList()
            ..sort();
      expect(bottom.isNotEmpty, isTrue);
      expect(bottom[bottom.length ~/ 2], greaterThan(388));
    },
  );

  test('pure-black receipt printing connected to the frame is not padding', () {
    final bytes = Uint8List(400 * 400 * 3)..fillRange(0, 400 * 400 * 3, 180);
    for (var y = 350; y < 400; y++) {
      for (var x = 60; x <= 340; x++) {
        for (var c = 0; c < 3; c++) {
          {
            bytes[(y * 400 + x) * 3 + c] = 0;
          }
        }
      }
    }
    final r = TrackerRaster(
      rgbBytes: bytes,
      width: 400,
      height: 400,
      rowStride: 1200,
    );
    final shape = [
      const BoundarySegment(TrackerPoint(.15, .2), TrackerPoint(.85, .2)),
      const BoundarySegment(TrackerPoint(.85, .2), TrackerPoint(.85, .99)),
      const BoundarySegment(TrackerPoint(.85, .99), TrackerPoint(.15, .99)),
      const BoundarySegment(TrackerPoint(.15, .99), TrackerPoint(.15, .2)),
    ];
    final result = constrainToImageCrop(r, shape);
    expect(result.any((s) => s.isImageClip), isFalse);
    expect(result[2].start.y, .99);
  });
}
