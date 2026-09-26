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
