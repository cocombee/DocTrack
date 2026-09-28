import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:pen_tool/flattening/flatten_image.dart';
import 'package:pen_tool/flattening/interior_mesh.dart';
import 'package:tracker_engine/tracker_engine.dart';

void main() {
  test(
    'reuse matches complete redraw across stock, alpha, windows and scale',
    () async {
      final source = img.Image(width: 96, height: 80, numChannels: 4);
      for (var y = 0; y < source.height; y++) {
        for (var x = 0; x < source.width; x++) {
          final ink =
              (y == 27 && x > 16 && x < 80) ||
              (y > 40 && y < 47 && x > 20 && x < 35);
          source.setPixelRgba(
            x,
            y,
            ink ? 30 : 175 + y ~/ 2,
            ink ? 25 : 185 + x ~/ 3,
            ink ? 35 : 180 + (x + y) ~/ 4,
            x < 9 && y < 15 ? 0 : (x > 86 && y < 12 ? 127 : 255),
          );
        }
      }
      final bytes = Uint8List.fromList(img.encodePng(source));
      const corners = [
        TrackerPoint(0, 0),
        TrackerPoint(1, 0),
        TrackerPoint(1, 1),
        TrackerPoint(0, 1),
      ];
      final boundary = BoundaryResult(
        width: 96,
        height: 80,
        status: BoundaryStatus.detected,
        outerBoundary: [],
        corners: corners,
        boundarySegments: [
          for (var i = 0; i < 4; i++)
            BoundarySegment(corners[i], corners[(i + 1) % 4]),
        ],
        confidence: 1,
        modelVersion: 'independent raster fixture',
        preprocessingVersion: 'upright',
      );
      final leaf = {
        'rows': 9,
        'columns': 9,
        'method': 'leaf',
        'grid': [
          for (var y = 0; y < 9; y++)
            for (var x = 0; x < 9; x++) [x / 8, y / 8],
        ],
      };
      final parent = {
        ...leaf,
        'method': 'first patch',
        'grid': [
          for (var y = 0; y < 9; y++)
            for (var x = 0; x < 9; x++)
              [x / 8, y / 8 + (x == 3 && y == 2 ? .025 : 0)],
        ],
        'preserved_mesh': leaf,
        'coordinate_transform': true,
        'refinement_regions': [
          [.125, .125, .625, .625],
        ],
      };
      for (final coordinate in [true, false]) {
        final mesh = InteriorMesh.fromJson({
          'rows': 5,
          'columns': 5,
          'method': 'edge and overlapping window',
          'grid_window': [9, 9, 0, 0],
          'grid': [
            for (var y = 0; y < 5; y++)
              for (var x = 0; x < 5; x++)
                [
                  x / 8 - (x == 0 && y == 0 ? .025 : 0),
                  y / 8 + (x == 2 && y == 2 ? .015 : 0),
                ],
          ],
          'coordinate_transform': coordinate,
          'preserved_mesh': parent,
          'refinement_regions': [
            [0, 0, .5, .5],
          ],
        });
        for (final settings in [
          (repair: false, rounded: false, scale: 1.0),
          (repair: true, rounded: false, scale: 1.0),
          (repair: true, rounded: true, scale: 1.0),
          (repair: true, rounded: false, scale: 1.05),
        ]) {
          Future<FlattenedImage> render(bool reuse) => flattenImage(
            bytes,
            boundary,
            interiorMesh: mesh,
            maxDimension: 96,
            estimatedRatio: 1.2,
            repairBorder: settings.repair,
            preserveRoundedCorners: settings.rounded,
            documentScale: settings.scale,
            reuseParentRaster: reuse,
          );
          final reference = await render(false);
          final reused = await render(true);
          expect(
            reused.png,
            reference.png,
            reason: 'coordinate=$coordinate, settings=$settings',
          );
          expect(reused.width, reference.width);
          expect(reused.height, reference.height);
        }
      }
    },
  );
}
