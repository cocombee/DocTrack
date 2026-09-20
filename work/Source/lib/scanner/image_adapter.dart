import 'dart:typed_data';
import 'dart:isolate';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';

Future<TrackerRaster> decodeUprightRaster(Uint8List encoded) => Isolate.run(() {
    final decoded=img.decodeImage(encoded);
    if(decoded==null)throw const FormatException('Unsupported or unreadable image');
    final upright=img.bakeOrientation(decoded);
    return TrackerRaster(rgbBytes:upright.getBytes(order:img.ChannelOrder.rgb),width:upright.width,height:upright.height,rowStride:upright.width*3);
  });

Future<BoundaryResult> detectEncodedImage(Uint8List encoded) async {
  final raster=await decodeUprightRaster(encoded);
  final engine=createTrackerEngine();
  try {await engine.initialize();return await engine.detectBoundary(raster);} finally {await engine.dispose();}
}
