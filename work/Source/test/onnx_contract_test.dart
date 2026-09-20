import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';
void main() {
 test('optional adapter rejects malformed rasters and respects disposal',() async {
  final engine=OnnxTracker(Uint8List.fromList([1]),Uint8List.fromList([1]));
  await engine.initialize();
  await expectLater(engine.detectBoundary(TrackerRaster(rgbBytes:Uint8List(3),width:8,height:8,rowStride:24)),throwsArgumentError);
  await engine.dispose();await engine.dispose();
  await expectLater(engine.initialize(),throwsStateError);
 });
 test('uniform raster is not falsely treated as a document',() async {
  final engine=OnnxTracker(Uint8List.fromList([1]),Uint8List.fromList([1]));
  final result=await engine.detectBoundary(TrackerRaster(rgbBytes:Uint8List(96*96*3),width:96,height:96,rowStride:288));
  expect(result.status,BoundaryStatus.notFound);expect(result.outerBoundary,isEmpty);await engine.dispose();
 });
 test('optional adapter rejects absent models before inference',() async {
  final engine=OnnxTracker(Uint8List(0),Uint8List(0));
  await expectLater(engine.initialize(),throwsArgumentError);await engine.dispose();
 });
}
