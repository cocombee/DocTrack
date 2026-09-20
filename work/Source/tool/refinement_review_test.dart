// ignore_for_file: avoid_print
import 'dart:io';
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';
void main(){test('evaluate candidate pixel refinement independently of labels',() {
 final manifest=jsonDecode(File('../Data/manifest.json').readAsStringSync()) as List;
 final out=Directory('../Scanner Integration/Refinement Review')..createSync(recursive:true);
 for(final e in [manifest[40]]) {
  final d=jsonDecode(File('../Scanner Integration/Predictions Simple V17/${e['name']}.json').readAsStringSync());
  final im=img.bakeOrientation(img.decodeImage(File('../Data/${e['image']}').readAsBytesSync())!);
  final raster=TrackerRaster(rgbBytes:im.getBytes(order:img.ChannelOrder.rgb),width:im.width,height:im.height,rowStride:im.width*3);
  final prior=(d['corners'] as List).map((p)=>TrackerPoint(p['x'],p['y'])).toList();
  final rigid=verifyRigidShape(raster,prior,searchScale:.10,diagnostic:(s)=>print('${e['name']}: $s'));
  final corners=rigid?.corners??refinePaperProposal(raster,prior);
  final segments=<BoundarySegment>[];
  final boundary=rigid==null?refineDocumentBoundary(raster,corners,roundCorners:false):[for(var i=0;i<4;i++)for(var j=0;j<80;j++)TrackerPoint(corners[i].x+(corners[(i+1)%4].x-corners[i].x)*j/80,corners[i].y+(corners[(i+1)%4].y-corners[i].y)*j/80)];
  segments.addAll(classifyPaperCorners(raster,rigid!=null?boundary:stabilizePaperSides(raster,bridgeLocalizedOcclusion(raster,refineFineEdges(boundary,raster),corners),corners),corners,rigid:rigid!=null));
  File('${out.path}/${e['name']}.json').writeAsStringSync(jsonEncode({'rigid':rigid!=null,'width':im.width,'height':im.height,'boundarySegments':segments.map((s)=>s.toJson()).toList()}));
 }
},timeout:const Timeout(Duration(minutes:3)));}
