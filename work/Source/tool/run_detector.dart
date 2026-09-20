import 'dart:convert';
import 'dart:io';
import 'package:pen_tool/scanner/image_adapter.dart';
Future<void> main(List<String> args) async {
  // This runner only receives an image directory. There is no manifest/label input.
  final files=Directory(args[0]).listSync(recursive:true).whereType<File>().where((f)=>RegExp(r'\.(jpg|jpeg|png)$',caseSensitive:false).hasMatch(f.path)).toList()..sort((a,b)=>a.path.compareTo(b.path));
  final output=Directory(args[1])..createSync(recursive:true);
  for(final file in files){final timer=Stopwatch()..start();try{final result=await detectEncodedImage(await file.readAsBytes());timer.stop();final json=result.toJson()..['elapsedMs']=timer.elapsedMilliseconds;File('${output.path}/${file.uri.pathSegments.last}.json').writeAsStringSync(jsonEncode(json));stdout.writeln('${file.uri.pathSegments.last}: ${result.status.name} ${timer.elapsedMilliseconds}ms');}catch(e){stderr.writeln('$file: $e');exitCode=1;}}
}
