import 'dart:typed_data';

import 'package:pen_tool/flattening/isolate_document.dart';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:pen_tool/flattening/flatten_geometry.dart';
import 'package:pen_tool/flattening/flatten_image.dart';
import 'package:pen_tool/flattening/document_format.dart';
import 'package:pen_tool/flattening/interior_mesh.dart';

void main() {
  test(
    'local correction preserves every outside pixel and the full silhouette',
    () async {
      final image = img.Image(width: 80, height: 80, numChannels: 4);
      for (var y = 0; y < 80; y++) {
        for (var x = 0; x < 80; x++) {
          image.setPixelRgba(
            x,
            y,
            180 + y ~/ 3,
            190 + x ~/ 3,
            170 + (x + y) ~/ 5,
            255,
          );
        }
      }
      const corners = [
        TrackerPoint(0, 0),
        TrackerPoint(1, 0),
        TrackerPoint(1, 1),
        TrackerPoint(0, 1),
      ];
      final boundary = BoundaryResult(
        width: 80,
        height: 80,
        status: BoundaryStatus.detected,
        outerBoundary: [],
        boundarySegments: [
          for (var i = 0; i < 4; i++)
            BoundarySegment(corners[i], corners[(i + 1) % 4]),
        ],
        corners: corners,
        confidence: 1,
        modelVersion: 'test',
        preprocessingVersion: 'test',
      );
      final original = InteriorMesh.identity(rows: 5, columns: 5);
      final correction = InteriorMesh.fromJson({
        'rows': 5,
        'columns': 5,
        'method': 'local',
        'grid': [
          for (var y = 0; y < 5; y++)
            for (var x = 0; x < 5; x++)
              [x / 4, y / 4 + (x == 2 && y == 2 ? .04 : 0)],
        ],
        'preserved_mesh': {
          'rows': 5,
          'columns': 5,
          'method': 'prior',
          'grid': [
            for (final point in original.points) [point.x, point.y],
          ],
        },
        'refinement_regions': [
          [.25, .25, .75, .75],
        ],
      });
      final bytes = Uint8List.fromList(img.encodePng(image));
      Future<img.Image> render(InteriorMesh mesh) async => img.decodePng(
        (await flattenImage(
          bytes,
          boundary,
          ratio: 1,
          maxDimension: 80,
          followCurves: false,
          interiorMesh: mesh,
        )).png,
      )!;
      final prior = await render(original), revised = await render(correction);
      final storedWindow = InteriorMesh.fromJson({
        'rows': 3,
        'columns': 3,
        'method': 'local window',
        'grid_window': [5, 5, 1, 1],
        'grid': [
          for (var y = 1; y <= 3; y++)
            for (var x = 1; x <= 3; x++)
              [correction.points[y * 5 + x].x, correction.points[y * 5 + x].y],
        ],
        'preserved_mesh': {
          'rows': 5,
          'columns': 5,
          'method': 'prior',
          'grid': [
            for (final p in original.points) [p.x, p.y],
          ],
        },
        'refinement_regions': [
          [.25, .25, .75, .75],
        ],
      });
      final compactRender = await render(storedWindow);
      expect(compactRender.getBytes(), revised.getBytes());
      var changedInside = 0;
      for (var y = 0; y < 80; y++) {
        for (var x = 0; x < 80; x++) {
          final a = prior.getPixel(x, y), b = revised.getPixel(x, y);
          expect(b.a, a.a);
          final changed = a.r != b.r || a.g != b.g || a.b != b.b;
          if (x < 20 || x >= 60 || y < 20 || y >= 60) {
            expect(changed, isFalse, reason: 'protected pixel $x,$y');
          } else if (changed) {
            changedInside++;
          }
        }
      }
      expect(changedInside, greaterThan(0));
      final second = InteriorMesh(
        correction.rows,
        correction.columns,
        [
          for (var i = 0; i < correction.points.length; i++)
            TrackerPoint(
              correction.points[i].x,
              correction.points[i].y + (i == 12 ? .04 : 0),
            ),
        ],
        'second local correction',
        preservedMesh: correction,
        refinementRegions: const [
          [.3, .3, .7, .7],
        ],
      );
      final twice = await render(second);
      var secondChanges = 0;
      for (var y = 0; y < 80; y++) {
        for (var x = 0; x < 80; x++) {
          final a = revised.getPixel(x, y), b = twice.getPixel(x, y);
          expect(b.a, a.a);
          final changed = a.r != b.r || a.g != b.g || a.b != b.b;
          if (x < 24 || x >= 56 || y < 24 || y >= 56) {
            expect(changed, isFalse, reason: 'prior refinement pixel $x,$y');
          } else if (changed) {
            secondChanges++;
          }
        }
      }
      expect(secondChanges, greaterThan(0));
    },
  );
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
  test(
    'separate projection angles preserve first crop pins and silhouette',
    () {
      final crop = q.toList(), references = q.toList();
      references[0] = const TrackerPoint(.18, .13);
      final segments = edges(crop);
      final g = FlattenGeometry.withProjectionFrame(
        crop,
        segments,
        800,
        600,
        referenceCorners: references,
      );
      expect(g.corners, q);
      expect(g.segments, same(segments));
      expect(g.followCurves, isFalse);
      final originalCrop = FlattenGeometry(
        q,
        segments,
        800,
        600,
        followCurves: false,
      );
      for (var y = 0; y <= 20; y++) {
        for (var x = 0; x <= 20; x++) {
          final point = TrackerPoint(x / 20, y / 20);
          expect(g.contains(point), originalCrop.contains(point));
        }
      }
      final outsideUnitFrame = g.rectify(q[0]);
      expect(outsideUnitFrame.x < 0 || outsideUnitFrame.y < 0, isTrue);
      for (var i = 0; i < 4; i++) {
        final source = g.source(i == 1 || i == 2 ? 1 : 0, i >= 2 ? 1 : 0);
        expect(source.x, closeTo(references[i].x, 1e-9));
        expect(source.y, closeTo(references[i].y, 1e-9));
      }
      crop[0] = const TrackerPoint(0, 0);
      references[0] = const TrackerPoint(1, 1);
      expect(g.corners, q);
      expect(g.projectionCorners[0].x, .18);
      expect(g.source(0, 0).x, closeTo(.18, 1e-9));
    },
  );
  test(
    'second frame sets camera proportions without a card size assumption',
    () {
      const reference = [
        TrackerPoint(.2, .2),
        TrackerPoint(.8, .2),
        TrackerPoint(.8, .7),
        TrackerPoint(.2, .7),
      ];
      final g = FlattenGeometry.withProjectionFrame(
        q,
        edges(q),
        800,
        600,
        referenceCorners: reference,
      );
      expect(g.estimatedRatio, closeTo(1.6, 1e-9));
      final point = g.rectify(g.source(.27, .64));
      expect(point.x, closeTo(.27, 1e-9));
      expect(point.y, closeTo(.64, 1e-9));
    },
  );
  test(
    'invalid second frame or crop-driven dewarp cannot masquerade as camera',
    () {
      expect(
        () => FlattenGeometry(q, edges(q), 800, 600, projectionCorners: q),
        throwsArgumentError,
      );
      for (final reference in [
        q.take(3).toList(),
        q.reversed.toList(),
        [q[0], q[2], q[1], q[3]],
        [const TrackerPoint(double.nan, 0), ...q.skip(1)],
      ]) {
        expect(
          () => FlattenGeometry.withProjectionFrame(
            q,
            edges(q),
            800,
            600,
            referenceCorners: reference,
          ),
          throwsArgumentError,
        );
      }
    },
  );
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
