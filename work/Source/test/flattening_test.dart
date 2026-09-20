import 'dart:typed_data';

import 'package:pen_tool/flattening/isolate_document.dart';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:pen_tool/flattening/flatten_geometry.dart';
import 'package:pen_tool/flattening/flatten_image.dart';
import 'package:pen_tool/flattening/document_format.dart';

void main() {
  const q = [
    TrackerPoint(.15, .1),
    TrackerPoint(.85, .2),
    TrackerPoint(.75, .85),
    TrackerPoint(.2, .8),
  ];
  List<BoundarySegment> edges(List<TrackerPoint> p) => [
    for (var i = 0; i < 4; i++) BoundarySegment(p[i], p[(i + 1) % 4]),
  ];
  test('perspective map exactly reproduces four source corners', () {
    final g = FlattenGeometry(q, edges(q), 800, 600, followCurves: false);
    for (var i = 0; i < 4; i++) {
      final p = g.source(i == 1 || i == 2 ? 1 : 0, i >= 2 ? 1 : 0);
      expect(p.x, closeTo(q[i].x, 1e-9));
      expect(p.y, closeTo(q[i].y, 1e-9));
    }
    final p = g.rectify(g.source(.37, .62));
    expect(p.x, closeTo(.37, 1e-9));
    expect(p.y, closeTo(.62, 1e-9));
  });
  test('bowed side generates a nonfolding interior mesh', () {
    const p = [
      TrackerPoint(.1, .1),
      TrackerPoint(.9, .1),
      TrackerPoint(.9, .9),
      TrackerPoint(.1, .9),
    ];
    final g = FlattenGeometry(
      p,
      [
        const BoundarySegment(
          TrackerPoint(.1, .1),
          TrackerPoint(.9, .1),
          control1: TrackerPoint(.3, .04),
          control2: TrackerPoint(.7, .04),
        ),
        ...edges(p).skip(1),
      ],
      1000,
      1000,
    );
    expect(g.source(.5, 0).y, closeTo(.055, .001));
    expect(g.source(.5, .5).y, lessThan(.5));
  });
  test(
    'format identification does not invent A4 versus B5 from aspect alone',
    () {
      expect(identifyDocumentFormat(1.414, []).label, contains('family'));
      expect(identifyDocumentFormat(1.414, ['B5']).label, 'B5');
      expect(
        identifyDocumentFormat(1.55, ['DRIVING LICENCE']).landscapeRatio,
        closeTo(1.5858, .001),
      );
      expect(identifyDocumentFormat(2.6, []).landscapeRatio, isNull);
    },
  );
  test(
    'rounded cutout stays transparent and source pixels are preserved',
    () async {
      const p = [
        TrackerPoint(.1, .1),
        TrackerPoint(.9, .1),
        TrackerPoint(.9, .9),
        TrackerPoint(.1, .9),
      ];
      final image = img.Image(width: 100, height: 100);
      img.fill(image, color: img.ColorRgb8(21, 80, 170));
      final segments = [
        const BoundarySegment(
          TrackerPoint(.1, .2),
          TrackerPoint(.2, .1),
          control1: TrackerPoint(.1, .13),
          control2: TrackerPoint(.13, .1),
        ),
        const BoundarySegment(TrackerPoint(.2, .1), TrackerPoint(.8, .1)),
        const BoundarySegment(
          TrackerPoint(.8, .1),
          TrackerPoint(.9, .2),
          control1: TrackerPoint(.87, .1),
          control2: TrackerPoint(.9, .13),
        ),
        const BoundarySegment(TrackerPoint(.9, .2), TrackerPoint(.9, .9)),
        edges(p)[2],
        const BoundarySegment(TrackerPoint(.1, .9), TrackerPoint(.1, .2)),
      ];
      final boundary = BoundaryResult(
        width: 100,
        height: 100,
        status: BoundaryStatus.detected,
        outerBoundary: [],
        boundarySegments: segments,
        corners: p,
        confidence: 1,
        modelVersion: 'test',
        preprocessingVersion: 'test',
      );
      final result = await flattenImage(
        Uint8List.fromList(img.encodePng(image)),
        boundary,
        ratio: 1,
        maxDimension: 80,
        // No recognised size or OCR: the detected corner family is sufficient.
      );
      final out = img.decodePng(result.png)!;
      expect(out.getPixel(0, 0).a, 0);
      expect(out.getPixel(40, 40).a, 255);
      expect(out.getPixel(40, 40).b, 170);
      final paper = img.decodePng(
        (await flattenImage(
          Uint8List.fromList(img.encodePng(image)),
          boundary,
          ratio: 1,
          maxDimension: 80,
          preserveRoundedCorners: false,
        )).png,
      )!;
      expect(
        paper.getPixel(0, 0).a,
        255,
        reason: 'Normal paper repairs corner cuts to a rectangle',
      );

      final receipt = img.decodePng(
        (await flattenImage(
          Uint8List.fromList(img.encodePng(image)),
          boundary,
          ratio: 1,
          maxDimension: 80,
          textLines: const ['Receipt', 'Total', '12.00', '20.00', '8.00'],
        )).png,
      )!;
      expect(
        receipt.getPixel(0, 0).a,
        255,
        reason: 'Confirmed transaction paper repairs cuts, not card rounding',
      );
      expect(image.width, 100);
    },
  );
  test(
    'edge repair preserves the original layout measurement pixels',
    () async {
      final image = img.Image(width: 200, height: 200, numChannels: 4);
      img.fill(image, color: img.ColorRgba8(25, 30, 35, 255));
      img.fillRect(
        image,
        x1: 20,
        y1: 20,
        x2: 179,
        y2: 179,
        color: img.ColorRgba8(220, 225, 230, 255),
      );
      for (var y = 25; y < 175; y++) {
        image.setPixelRgba(20, y, 150, 155, 160, 255);
      }
      const p = [
        TrackerPoint(.1, .1),
        TrackerPoint(.9, .1),
        TrackerPoint(.9, .9),
        TrackerPoint(.1, .9),
      ];
      final boundary = BoundaryResult(
        width: 200,
        height: 200,
        status: BoundaryStatus.detected,
        outerBoundary: [],
        boundarySegments: edges(p),
        corners: p,
        confidence: 1,
        modelVersion: 'test',
        preprocessingVersion: 'test',
      );
      final bytes = Uint8List.fromList(img.encodePng(image));
      final old = await isolateDocument(bytes, boundary, coherentEdges: false);
      final candidate = await isolateDocument(
        bytes,
        boundary,
        coherentEdges: true,
      );
      expect(candidate.measurementPng, orderedEquals(old.png));
      final selected = img.decodePng(candidate.png)!;
      final measured = img.decodePng(candidate.measurementPng!)!;
      expect(selected.getPixel(0, 80).a, 0);
      expect(measured.getPixel(0, 80).a, 255);
      expect(selected.getPixel(1, 80).a, 255);
    },
  );
  test('isolation preserves an existing transparent source pixel', () async {
    final image = img.Image(width: 100, height: 100, numChannels: 4);
    img.fill(image, color: img.ColorRgba8(220, 225, 230, 255));
    image.setPixelRgba(50, 50, 0, 0, 0, 0);
    final boundary = BoundaryResult(
      width: 100,
      height: 100,
      status: BoundaryStatus.detected,
      outerBoundary: [],
      boundarySegments: edges(q),
      corners: q,
      confidence: 1,
      modelVersion: 'test',
      preprocessingVersion: 'test',
    );
    final selected = await isolateDocument(
      Uint8List.fromList(img.encodePng(image)),
      boundary,
    );
    final out = img.decodePng(selected.png)!;
    // The selected bounding box begins at source pixel (15,10).
    expect(out.getPixel(35, 40).a, 0);
    expect(out.getPixel(36, 40).a, 255);
  });
  test(
    'paper isolation preserves pixels and final repair has four straight sides',
    () async {
      const p = [
        TrackerPoint(.1, .1),
        TrackerPoint(.9, .1),
        TrackerPoint(.9, .9),
        TrackerPoint(.1, .9),
      ];
      final image = img.Image(width: 100, height: 100);
      img.fill(image, color: img.ColorRgb8(210, 220, 230));
      image.setPixelRgb(50, 50, 10, 20, 30);
      final boundary = BoundaryResult(
        width: 100,
        height: 100,
        status: BoundaryStatus.detected,
        outerBoundary: [],
        corners: p,
        confidence: 1,
        modelVersion: 'test',
        preprocessingVersion: 'test',
        boundarySegments: [
          const BoundarySegment(
            TrackerPoint(.1, .1),
            TrackerPoint(.9, .1),
            control1: TrackerPoint(.3, .16),
            control2: TrackerPoint(.7, .16),
          ),
          const BoundarySegment(TrackerPoint(.9, .1), TrackerPoint(.9, .9)),
          const BoundarySegment(TrackerPoint(.9, .9), TrackerPoint(.1, .9)),
          const BoundarySegment(TrackerPoint(.1, .9), TrackerPoint(.1, .6)),
          const BoundarySegment(TrackerPoint(.1, .6), TrackerPoint(.2, .5)),
          const BoundarySegment(TrackerPoint(.2, .5), TrackerPoint(.1, .4)),
          const BoundarySegment(TrackerPoint(.1, .4), TrackerPoint(.1, .1)),
        ],
      );
      final selected = await isolateDocument(
        Uint8List.fromList(img.encodePng(image)),
        boundary,
      );
      final cut = img.decodePng(selected.png)!;
      expect(cut.width, 80);
      expect(cut.getPixel(40, 40).r, 10);
      expect(cut.getPixel(0, 40).a, 0);
      final out = img.decodePng(
        (await flattenImage(
          selected.png,
          selected.boundary,
          ratio: 1,
          maxDimension: 80,
          estimatedRatio: selected.estimatedRatio,
        )).png,
      )!;
      for (var i = 0; i < 80; i++) {
        for (final pixel in [
          out.getPixel(i, 0),
          out.getPixel(i, 79),
          out.getPixel(0, i),
          out.getPixel(79, i),
        ]) {
          expect(pixel.a, 255);
          expect(
            pixel.r,
            greaterThan(180),
            reason: 'No black background fringe',
          );
        }
      }
    },
  );
  test('degenerate shape is rejected', () {
    expect(
      () => FlattenGeometry(
        List.filled(4, const TrackerPoint(.1, .1)),
        edges(q),
        100,
        100,
      ),
      throwsArgumentError,
    );
  });
}
