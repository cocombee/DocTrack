// ignore_for_file: avoid_print
import 'dart:io';
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';
void main(){test('inspect shape hypotheses',() async {
 final manifest=jsonDecode(File('../Data/manifest.json').readAsStringSync()) as List;
 for(final index in [5,6,35,47]) {
  final e=manifest[index-1],file=File('../Scanner Integration/Predictions Targeted V16/${e['name']}.json');
  final d=jsonDecode(file.readAsStringSync());
  final im=img.bakeOrientation(img.decodeImage(File('../Data/${e['image']}').readAsBytesSync())!);
  final shape=reconstructDocumentShape(TrackerRaster(rgbBytes:im.getBytes(order:img.ChannelOrder.rgb),width:im.width,height:im.height,rowStride:im.width*3),(d['corners'] as List).map((p)=>TrackerPoint(p['x'],p['y'])).toList(),coarseOnly:true);
  print('$index: ${shape.reasons}');
  final folder=Directory('../Scanner Integration/Coarse Shape Trial')..createSync(recursive:true);
  File('${folder.path}/${e['name']}.json').writeAsStringSync(jsonEncode({'boundarySegments':shape.segments.map((s)=>s.toJson()).toList()}));
 }
});}
