import 'dart:isolate';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

import 'flatten_image.dart';

class DocumentOrientationJob {
  static int _nextId = 1000000;
  final int id = ++_nextId;
  bool cancelled = false;
  void cancel() {
    cancelled = true;
    const MethodChannel('doctrack/window')
        .invokeMethod<void>('cancelDocumentText', id)
        .catchError((Object _) {});
  }

  Future<FlattenedImage> run(FlattenedImage input) async {
    try {
      final answer = await const MethodChannel('doctrack/window')
          .invokeMapMethod<String, dynamic>('orientFlattenedDocument', {
            'data': input.png,
            'id': id,
          });
      if (cancelled) return input;
      final degrees = answer?['degrees'];
      if (degrees is! int || ![90, 180, 270].contains(degrees)) return input;
      return await rotateFlatImage(input, degrees);
    } on MissingPluginException {
      return input;
    } on PlatformException {
      return input;
    }
  }
}

Future<FlattenedImage> rotateFlatImage(FlattenedImage input, int degrees) =>
    Isolate.run(() {
      if (![0, 90, 180, 270].contains(degrees)) {
        throw ArgumentError('Quarter turns only');
      }
      if (degrees == 0) return input;
      final image = img.decodePng(input.png);
      if (image == null) throw const FormatException('Unreadable flat image');
      final rotated = img.copyRotate(image, angle: degrees);
      return FlattenedImage(
        Uint8List.fromList(img.encodePng(rotated)),
        rotated.width,
        rotated.height,
        rotated.width / rotated.height,
        input.format,
      );
    });
