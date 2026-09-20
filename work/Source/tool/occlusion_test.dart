import 'dart:io';
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';
void main(){test('occlusion diagnostics',(){
 for(final name in ['idcard_final_polygon_004383.jpg','idcard_final_polygon_005361.jpg']){
 final f=Directory('../Data/originals').listSync(recursive:true).whereType<File>().firstWhere((f)=>f.path.endsWith(name));
 final im=img.decodeImage(f.readAsBytesSync())!;
 final d=jsonDecode(File('../Scanner Integration/Predictions Dart ONNX V13/$name.json').readAsStringSync());
 stdout.writeln(name);reconstructOccludedCard(TrackerRaster(rgbBytes:im.getBytes(order:img.ChannelOrder.rgb),width:im.width,height:im.height,rowStride:im.width*3),[for(final v in d['corners'])TrackerPoint(v['x'],v['y'])],diagnostic:stdout.writeln);
 }
});}
