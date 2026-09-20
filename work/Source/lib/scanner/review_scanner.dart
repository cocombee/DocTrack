import 'dart:io';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';
import 'image_adapter.dart';

// DocTrack research host only. Akaunting must supply approved bundled model
// bytes itself; the reusable engines have no hardcoded paths or downloads.
const _models='/Users/mrkokom/My Drive (cocombees@gmail.com)/Codex/Apps/DocTrack/work/Scanner Integration/DocAligner Evaluation';
class ReviewScanCancellation {
  static int _serial = 0;
  final int id = ++_serial;
  bool cancelled = false;
  OnnxTracker? engine;
  void cancel() { cancelled = true; engine?.cancelDetection(); const MethodChannel('doctrack/window').invokeMethod<void>('cancelDocumentText',id).catchError((Object _) {}); }
  void check() { if (cancelled) throw StateError('Detection cancelled'); }
}
Future<BoundaryResult> detectReviewImage(Uint8List encoded, {ReviewScanCancellation? cancellation}) async {
  cancellation?.check();
  final decoded=await decodeUprightRaster(encoded);
  cancellation?.check();
  var kind=DocumentKind.unknown;
  try {
    final lines=await const MethodChannel('doctrack/window').invokeListMethod<String>('recognizeDocumentText',{'data':encoded,'id':cancellation?.id ?? 0});
    final evidence=DocumentContentEvidence.fromLines(lines ?? []);
    kind=evidence.kind;
    if (const bool.fromEnvironment('DOCTRACK_DIAGNOSTICS')) {
      await File('/tmp/doctrack-content-status.json').writeAsString(jsonEncode({'lines':lines?.length ?? 0,'kind':kind.name,'identityTerms':evidence.identityTerms,'transactionTerms':evidence.transactionTerms,'amountRows':evidence.amountRows}));
    }
  } on MissingPluginException { /* Other hosts provide their own text adapter. */ }
    on PlatformException { /* Unreadable content remains unknown. */ }
  final raster=TrackerRaster(rgbBytes:decoded.rgbBytes,width:decoded.width,height:decoded.height,rowStride:decoded.rowStride,validImagePolygon:decoded.validImagePolygon,documentKind:kind);
  cancellation?.check();
  final heat=File('$_models/lcnet100.onnx'),point=File('$_models/lcnet050.onnx');
  final TrackerEngine engine=await heat.exists()&&await point.exists()
      ? OnnxTracker(await heat.readAsBytes(),await point.readAsBytes())
      : createTrackerEngine();
  if (engine is OnnxTracker) cancellation?.engine = engine;
  cancellation?.check();
  try {await engine.initialize();cancellation?.check();return await engine.detectBoundary(raster);}finally{await engine.dispose();}
}
