import 'dart:convert';
import 'dart:io';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/src/classical.dart';
Future<void> main(List<String> args) async {
  final output=Directory(args[1])..createSync(recursive:true);
  for(final file in Directory(args[0]).listSync(recursive:true).whereType<File>().where((f)=>f.path.endsWith('.jpg'))){
    final im=img.bakeOrientation(img.decodeImage(await file.readAsBytes())!);final candidates=<Map<String,dynamic>>[];
    BoundaryEngine.detect(UprightRaster(im.getBytes(order:img.ChannelOrder.rgb),im.width,im.height,im.width*3),diagnostics:candidates);
    File('${output.path}/${file.uri.pathSegments.last}.json').writeAsStringSync(jsonEncode(candidates));
  }
}
