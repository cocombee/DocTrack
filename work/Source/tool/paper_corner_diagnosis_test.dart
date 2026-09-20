import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  test('trace paper corner stages without changing predictions', () {
    final entries =
        jsonDecode(File('../Data/manifest.json').readAsStringSync()) as List;
    final entry = entries[27];
    final image = img.bakeOrientation(
      img.decodeImage(File('../Data/${entry['image']}').readAsBytesSync())!,
    );
    final raster = TrackerRaster(
      rgbBytes: image.getBytes(order: img.ChannelOrder.rgb),
      width: image.width,
      height: image.height,
      rowStride: image.width * 3,
    );
    final old = jsonDecode(
      File(
        '../Scanner Integration/Predictions Dart ONNX V1/${entry['name']}.json',
      ).readAsStringSync(),
    );
    final prior = (old['corners'] as List)
        .map((p) => TrackerPoint(p['x'], p['y']))
        .toList();
    void report(String stage, List<TrackerPoint> corners) => stdout.writeln(
      '$stage: ${corners.map((p) => [p.x * image.width, p.y * image.height]).toList()}',
    );
    report('Neural prior', prior);
    final rigid = verifyRigidShape(raster, prior, diagnostic: stdout.writeln);
    expect(rigid, isNotNull);
    report('Rigid intersection', rigid!.corners);
    final smooth = refineSmoothVerifiedSides(raster, rigid.corners);
    expect(smooth, isNotNull);
    report('Fine side result', smooth!.corners);
    final wide = refineSmoothVerifiedSides(
      raster,
      rigid.corners,
      materialRecovery: true,
    )!;
    stdout.writeln(
      'Material before ${List.generate(4, (i) => paperCornerMaterialMismatch(raster, smooth.corners, i))} after ${List.generate(4, (i) => paperCornerMaterialMismatch(raster, wide.corners, i))}',
    );
    final finalShape = refineVerifiedPaper(raster, prior);
    report('Final engine result', finalShape.corners);
    stdout.writeln('Reasons: ${finalShape.reasons}');
    expect(finalShape.reasons, contains('PAPER_CORNER_MATERIAL_RECOVERY'));
    expect(finalShape.corners[2].y * image.height, inInclusiveRange(342, 347));
    expect(
      paperCornerMaterialMismatch(raster, finalShape.corners, 2),
      lessThan(.5),
    );
  });
}
