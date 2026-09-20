import 'dart:io';
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';
void main(){test('audit foreground and outside-pattern evidence',(){
 final manifest=jsonDecode(File('../Data/manifest.json').readAsStringSync()) as List;
 final results=<Map<String,dynamic>>[];
 for(var index=0;index<manifest.length;index++) {
  final e=manifest[index],d=jsonDecode(File('../Scanner Integration/Predictions Targeted V16/${e['name']}.json').readAsStringSync());
  final im=img.bakeOrientation(img.decodeImage(File('../Data/${e['image']}').readAsBytesSync())!);
  TrackerPoint p(dynamic v)=>TrackerPoint((v['x'] as num).toDouble(),(v['y'] as num).toDouble());
  final segments=(d['boundarySegments'] as List).map((s)=>BoundarySegment(p(s['start']),p(s['end']),control1:s['control1']==null?null:p(s['control1']),control2:s['control2']==null?null:p(s['control2']))).toList();
  final conflict=boundaryMaterialConflict(TrackerRaster(rgbBytes:im.getBytes(order:img.ChannelOrder.rgb),width:im.width,height:im.height,rowStride:im.width*3),(d['corners'] as List).map(p).toList(),segments);
  results.add({'index':index+1,'conflict':conflict});
 }
 File('../Scanner Integration/Material Audit.json').writeAsStringSync(jsonEncode(results));
});}
