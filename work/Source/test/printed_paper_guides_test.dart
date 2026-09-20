import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

({TrackerRaster raster, List<TrackerPoint> corners}) printedSheet({
  int bentLines = 0,
}) {
  const size = 400;
  TrackerPoint project(double u, double v) => TrackerPoint(
    (.7 * u + .1 * v + .1) / (1 + .2 * v),
    (.05 * u + .75 * v + .1) / (1 + .2 * v),
  );
  final bytes = Uint8List(size * size * 3);
  for (var y = 0; y < size; y++) {
    for (var x = 0; x < size; x++) {
      final px = x / size, py = y / size;
      final b = .1 - .2 * px, e = .75 - .2 * py;
      final det = .7 * e - b * .05;
      final u = ((px - .1) * e - b * (py - .1)) / det;
      final v = (.7 * (py - .1) - .05 * (px - .1)) / det;
      final inside = u >= 0 && u <= 1 && v >= 0 && v <= 1;
      var ink = false;
      for (var line = 0; line < 4; line++) {
        final bend = line < bentLines && v > .5 ? .14 * (v - .5) : 0.0;
        if ((u - (.18 + .19 * line + bend)).abs() < .003 &&
            v > .06 &&
            v < .96) {
          ink = true;
        }
      }
      for (var c = 0; c < 3; c++) {
        bytes[(y * size + x) * 3 + c] = inside
            ? ink
                  ? 35
                  : 225
            : 70;
      }
    }
  }
  return (
    raster: TrackerRaster(
      rgbBytes: bytes,
      width: size,
      height: size,
      rowStride: size * 3,
    ),
    corners: [project(0, 0), project(1, 0), project(1, 1), project(0, 1)],
  );
}

void main() {
  test('straight printed rules remain straight under perspective', () {
    final sheet = printedSheet();
    final result = inspectPrintedPaperGuides(sheet.raster, sheet.corners);
    expect(result.guides, greaterThanOrEqualTo(3));
    expect(result.hasFold, isFalse);
  });
  test('several printed rules changing direction corroborate a fold', () {
    final sheet = printedSheet(bentLines: 4);
    final result = inspectPrintedPaperGuides(sheet.raster, sheet.corners);
    expect(result.hasFold, isTrue);
    expect(result.foldPosition, closeTo(.5, .08));
  });
  test('one irregular printed line does not authorize a paper fold', () {
    final sheet = printedSheet(bentLines: 1);
    expect(
      inspectPrintedPaperGuides(sheet.raster, sheet.corners).hasFold,
      isFalse,
    );
  });
}
