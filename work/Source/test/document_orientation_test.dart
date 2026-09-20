import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;

import 'package:pen_tool/flattening/document_orientation.dart';
import 'package:pen_tool/flattening/flatten_image.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('doctrack/window');
  final raster = img.Image(width: 3, height: 2, numChannels: 4);
  raster.setPixelRgba(0, 0, 10, 20, 30, 255);
  raster.setPixelRgba(2, 1, 80, 90, 100, 128);
  final input = FlattenedImage(
    Uint8List.fromList(img.encodePng(raster)),
    3,
    2,
    1.5,
    'Paper',
  );
  tearDown(
    () => TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null),
  );
  test(
    'quarter turn preserves transparent exterior and exact pixels',
    () async {
      final output = await rotateFlatImage(input, 90);
      expect([output.width, output.height], [2, 3]);
      final decoded = img.decodePng(output.png)!;
      expect(decoded.getPixel(1, 0).a, 255);
      expect(decoded.getPixel(1, 0).r, 10);
      expect(decoded.getPixel(0, 2).a, 128);
      expect(decoded.getPixel(0, 0).a, 0);
      final restored = await rotateFlatImage(output, 270);
      expect(img.decodePng(restored.png)!.getBytes(), raster.getBytes());
    },
  );
  test('ambiguous result preserves original image', () async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(
          channel,
          (_) async => {'degrees': 0, 'evidence': 'ambiguous'},
        );
    expect(await DocumentOrientationJob().run(input), same(input));
  });
  test('cancelled orientation cannot replace image', () async {
    final job = DocumentOrientationJob();
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
          if (call.method == 'orientFlattenedDocument') {
            job.cancel();
            return {'degrees': 90};
          }
          return null;
        });
    expect(await job.run(input), same(input));
  });
}
