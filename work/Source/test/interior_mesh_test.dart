import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/flattening/interior_mesh.dart';

Map<String, dynamic> mesh(double displacement) => {
  'rows': 5,
  'columns': 5,
  'method': 'test',
  'grid': [
    for (var y = 0; y < 5; y++)
      for (var x = 0; x < 5; x++)
        [x / 4, y / 4 + (x == 2 && y == 2 ? displacement : 0)],
  ],
};

void main() {
  test('coordinate changes compose through the retained nonlinear map', () {
    final data = {
      ...mesh(.03),
      'coordinate_transform': true,
      'preserved_mesh': mesh(.02),
      'refinement_regions': [
        [.25, .25, .75, .75],
      ],
    };
    final composed = InteriorMesh.fromJson(data);
    expect(composed.at(.5, .5).y, closeTo(.5476, 1e-14));
    final parent = InteriorMesh.fromJson(mesh(.02));
    for (final point in [
      [.1234567, .8765432],
      [.5, .1],
      [1.0, 1.0],
    ]) {
      expect(
        composed.at(point[0], point[1]).x,
        parent.at(point[0], point[1]).x,
      );
      expect(
        composed.at(point[0], point[1]).y,
        parent.at(point[0], point[1]).y,
      );
    }
    final unchanged = InteriorMesh.fromJson({...data, ...mesh(0)});
    expect(unchanged.at(.3456789, .4567891).y, parent.at(.3456789, .4567891).y);
  });
  test('malformed coordinate transforms cannot bypass validation', () {
    expect(
      () => InteriorMesh.fromJson({
        'rows': 2,
        'columns': 2,
        'method': 'fold at opposite corner',
        'grid': [
          [0, 0],
          [1, 0],
          [0, 1],
          [1, -.1],
        ],
        'coordinate_transform': true,
        'preserved_mesh': mesh(0),
        'refinement_regions': [
          [.1, .1, .9, .9],
        ],
      }),
      throwsFormatException,
    );
    expect(
      () => InteriorMesh.fromJson({...mesh(0), 'coordinate_transform': true}),
      throwsFormatException,
    );
    expect(
      () => InteriorMesh.fromJson({
        ...mesh(0),
        'coordinate_transform': 'true',
        'preserved_mesh': mesh(0),
        'refinement_regions': [
          [.25, .25, .75, .75],
        ],
      }),
      throwsFormatException,
    );
  });
  test('stored windows sample exactly like the original full grid', () {
    final data = {
      ...mesh(.03),
      'preserved_mesh': mesh(.01234567890123456),
      'refinement_regions': [
        [.25, .25, .75, .75],
      ],
    };
    final dense = InteriorMesh.fromJson(data);
    final cropped = InteriorMesh.fromJson({
      ...data,
      'rows': 3,
      'columns': 3,
      'grid_window': [5, 5, 1, 1],
      'grid': [
        for (var y = 1; y <= 3; y++)
          for (var x = 1; x <= 3; x++) (data['grid'] as List)[y * 5 + x],
      ],
    });
    for (var y = 0; y <= 100; y++) {
      for (var x = 0; x <= 100; x++) {
        final a = dense.at(x / 100, y / 100), b = cropped.at(x / 100, y / 100);
        expect(b.x, a.x, reason: 'x at $x,$y');
        expect(b.y, a.y, reason: 'y at $x,$y');
      }
    }
  });
  test('serialized node boundary keeps exact grid coverage', () {
    const bottom = .20875684128225178;
    expect(bottom * 5116, 1068);
    expect(bottom, greaterThan(1068 / 5116));
    final data = {
      'rows': 2,
      'columns': 2,
      'method': 'Measured local coordinate change',
      'coordinate_transform': true,
      'grid_window': [5117, 2, 1067, 0],
      'grid': [
        [0.0, 1067 / 5116],
        [1.0, 1067 / 5116],
        [0.0, 1068 / 5116],
        [1.0, 1068 / 5116],
      ],
      'preserved_mesh': mesh(0),
      'refinement_regions': [
        [0.0, 1067 / 5116, 1.0, bottom],
      ],
    };
    final local = InteriorMesh.fromJson(data);
    final parent = InteriorMesh.fromJson(mesh(0));
    for (final y in [1067 / 5116, 1067.5 / 5116, bottom]) {
      expect(local.at(.4, y).x, parent.at(.4, y).x);
      expect(local.at(.4, y).y, parent.at(.4, y).y);
    }
    expect(
      () => InteriorMesh.fromJson({
        ...data,
        'refinement_regions': [
          [0.0, 1067 / 5116, 1.0, bottom + .01 / 5116],
        ],
      }),
      throwsFormatException,
    );
  });
  test('invalid stored windows and uncovered patches are refused', () {
    for (final window in [
      [5, 5, -1, 1],
      [5, 5, 3, 1],
      [5, 5, 1],
      [5, 5, 1.5, 1],
    ]) {
      expect(
        () => InteriorMesh.fromJson({
          ...mesh(0),
          'grid_window': window,
          'preserved_mesh': mesh(0),
          'refinement_regions': [
            [.25, .25, .75, .75],
          ],
        }),
        throwsFormatException,
      );
    }
    expect(
      () => InteriorMesh.fromJson({
        ...mesh(0),
        'grid_window': [9, 9, 2, 2],
        'preserved_mesh': mesh(0),
        'refinement_regions': [
          [.1, .1, .9, .9],
        ],
      }),
      throwsFormatException,
    );
  });
  test('local meshes retain the exact prior map outside measured regions', () {
    final original = mesh(.01234567890123456);
    final corrected = InteriorMesh.fromJson({
      ...mesh(.03),
      'preserved_mesh': original,
      'refinement_regions': [
        [.25, .25, .75, .75],
      ],
    });
    final prior = InteriorMesh.fromJson(original);
    for (final point in [
      [.1234567, .8765432],
      [0.0, 0.0],
      [1.0, 1.0],
      [.5, .1],
    ]) {
      expect(
        corrected.at(point[0], point[1]).x,
        prior.at(point[0], point[1]).x,
      );
      expect(
        corrected.at(point[0], point[1]).y,
        prior.at(point[0], point[1]).y,
      );
    }
    expect(corrected.at(.5, .5).y, .53);
  });
  test(
    'invalid patch bounds and folded proposals cannot bypass validation',
    () {
      expect(
        () => InteriorMesh.fromJson({
          ...mesh(.03),
          'preserved_mesh': mesh(0),
          'refinement_regions': [
            [.75, .25, .25, .75],
          ],
        }),
        throwsFormatException,
      );
      expect(
        () => InteriorMesh.fromJson({
          ...mesh(.8),
          'preserved_mesh': mesh(0),
          'refinement_regions': [
            [.25, .25, .75, .75],
          ],
        }),
        throwsFormatException,
      );
    },
  );
}
