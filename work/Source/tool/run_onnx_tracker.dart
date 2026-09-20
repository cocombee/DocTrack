import 'dart:io';
import 'dart:convert';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';
Future<void> main(List<String> args) async {
 final textLines=args.length>4 && args[4].isNotEmpty ? jsonDecode(File(args[4]).readAsStringSync()) as Map : <String,dynamic>{};
 final selectedNames=args.length>5 ? (jsonDecode(File(args[5]).readAsStringSync()) as List).map((e)=>e['name'] as String).toSet() : null;
 final engine=OnnxTracker(await File(args[1]).readAsBytes(),await File(args[2]).readAsBytes());await engine.initialize();final out=Directory(args[3])..createSync(recursive:true);
 try {for(final f in Directory(args[0]).listSync(recursive:true).whereType<File>().where((v)=>v.path.endsWith('.jpg'))) {
  if(selectedNames!=null && !selectedNames.contains(f.uri.pathSegments.last)) continue;
  final clock=Stopwatch()..start();final im=img.bakeOrientation(img.decodeImage(await f.readAsBytes())!);final result=await engine.detectBoundary(TrackerRaster(rgbBytes:im.getBytes(order:img.ChannelOrder.rgb),width:im.width,height:im.height,rowStride:im.width*3,documentKind:DocumentContentEvidence.fromLines((textLines[f.uri.pathSegments.last] as List? ?? []).cast<String>()).kind));
  File('${out.path}/${f.uri.pathSegments.last}.json').writeAsStringSync(jsonEncode(result.toJson()..['elapsedMs']=clock.elapsedMilliseconds));stdout.writeln(f.uri.pathSegments.last);
 }}finally{await engine.dispose();}
}
