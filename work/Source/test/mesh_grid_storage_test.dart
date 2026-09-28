import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/flattening/interior_mesh.dart';

// Fixtures encoded independently with Python struct.pack('<dd') and gzip.
const identityBytes = 'H4sIAAAAAAACE2NgAIMGBhTwwZ4BK4CJI2gAhrQ8mkAAAAA=';
const foldedBytes = 'H4sIAAAAAAACE2NgwAY+2DPgFf9gP2smCOzcDwBMfailQAAAAA==';
const windowBytes =
    'H4sIAAAAAAACE2NgAIEL9gwo9AM0/gs0Prq6B/YeD6tE1rk/RFOPy5wHaHwEDQABeHmdkAAAAA==';

Map<String, dynamic> packed(String bytes) => {
  'rows': 2,
  'columns': 2,
  'method': 'independent storage fixture',
  'grid_encoding': 'float64le-gzip-base64',
  'grid_data': bytes,
};

int bits(double value) => (ByteData(
  8,
)..setFloat64(0, value, Endian.little)).getUint64(0, Endian.little);

void main() {
  test(
    'independent float payload preserves all bits, including negative zero',
    () {
      final encoded = InteriorMesh.fromJson(packed(identityBytes));
      final legacy = InteriorMesh.fromJson({
        'rows': 2,
        'columns': 2,
        'method': 'legacy',
        'grid': [
          [-0.0, 0.0],
          [1.0, 0.0],
          [0.0, 1.0],
          [1.0, 1.0],
        ],
      });
      for (var i = 0; i < legacy.points.length; i++) {
        expect(bits(encoded.points[i].x), bits(legacy.points[i].x));
        expect(bits(encoded.points[i].y), bits(legacy.points[i].y));
      }
      expect(encoded.evidence.containsKey('grid_data'), isFalse);
    },
  );

  test('compressed local coordinate windows keep exact parent sampling', () {
    final parent = {
      'rows': 5,
      'columns': 5,
      'method': 'nonlinear parent',
      'grid': [
        for (var y = 0; y < 5; y++)
          for (var x = 0; x < 5; x++)
            [x / 4, y / 4 + (x == 2 && y == 2 ? .02 : 0)],
      ],
    };
    final metadata = {
      'rows': 3,
      'columns': 3,
      'method': 'local coordinate change',
      'grid_window': [5, 5, 1, 1],
      'coordinate_transform': true,
      'preserved_mesh': parent,
      'refinement_regions': [
        [.25, .25, .75, .75],
      ],
    };
    final encoded = InteriorMesh.fromJson({
      ...metadata,
      'grid_encoding': 'float64le-gzip-base64',
      'grid_data': windowBytes,
    });
    final legacy = InteriorMesh.fromJson({
      ...metadata,
      'grid': [
        for (var y = 1; y < 4; y++)
          for (var x = 1; x < 4; x++)
            [x / 4, y / 4 + (x == 2 && y == 2 ? .04 : 0)],
      ],
    });
    for (final point in [
      [.5, .5],
      [.2499999, .3456789],
      [.2500001, .3456789],
      [.4567891, .6543219],
      [.75, .75],
      [.7500001, .75],
      [1.0, 1.0],
    ]) {
      final a = encoded.at(point[0], point[1]);
      final b = legacy.at(point[0], point[1]);
      expect(bits(a.x), bits(b.x));
      expect(bits(a.y), bits(b.y));
    }
  });

  test('compressed coordinates cannot bypass the existing fold guard', () {
    expect(
      () => InteriorMesh.fromJson({
        ...packed(foldedBytes),
        'coordinate_transform': true,
        'preserved_mesh': packed(identityBytes),
        'refinement_regions': [
          [.1, .1, .9, .9],
        ],
      }),
      throwsFormatException,
    );
  });

  test('invalid, ambiguous and incorrectly sized payloads fail closed', () {
    for (final data in [
      {...packed(identityBytes), 'grid_encoding': 'float32'},
      {...packed(identityBytes), 'grid_data': 'not base64'},
      {...packed(identityBytes), 'rows': 3},
      {...packed(identityBytes), 'columns': 1},
      {...packed(identityBytes), 'rows': 1 << 30},
      {...packed(identityBytes), 'rows': 1 << 62},
      {...packed(identityBytes), 'columns': 1 << 62},
      {...packed(identityBytes), 'grid': []},
      {...packed(windowBytes)},
    ]) {
      expect(() => InteriorMesh.fromJson(data), throwsFormatException);
    }
  });
}
