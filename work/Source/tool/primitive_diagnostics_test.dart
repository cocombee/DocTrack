// ignore_for_file: avoid_print
import 'dart:io';
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';
void main(){test('diagnose visible-side hypotheses',(){
 final manifest=jsonDecode(File('../Data/manifest.json').readAsStringSync()) as List;
 for(final index in [3,22,24,35,46]) {
 final e=manifest[index];final im=img.bakeOrientation(img.decodeImage(File('../Data/${e['image']}').readAsBytesSync())!);
 final r=TrackerRaster(rgbBytes:im.getBytes(order:img.ChannelOrder.rgb),width:im.width,height:im.height,rowStride:im.width*3);
 final d=jsonDecode(File('../Scanner Integration/Predictions Physical V18/${e['name']}.json').readAsStringSync());
 final q=(d['corners'] as List).map((p)=>TrackerPoint(p['x'],p['y'])).toList();
 print('IMAGE ${index+1} reasons=${d['reasonCodes']}');
 final result=verifyRigidShape(r,q,diagnostic:print);
 print('RECONSTRUCTION $result');
 }
});}
