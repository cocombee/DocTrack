import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/scanner/detected_text_cache.dart';
import 'package:pen_tool/flattening/document_words.dart';

void main() {
  test('detection text is reused only for identical photo bytes', () {
    final cache = DetectedTextCache();
    final original = Uint8List.fromList([1, 2, 3]);
    final lines = ['GẶP', '935,000'];
    cache.record('receipt.jpg', original, lines);
    original[0] = 9;
    lines[0] = 'changed';

    expect(cache.lookup('receipt.jpg', Uint8List.fromList([1, 2, 3])), [
      'GẶP',
      '935,000',
    ]);
    expect(cache.lookup('receipt.jpg', original), isNull);
    expect(
      cache.lookup('different.jpg', Uint8List.fromList([1, 2, 3])),
      isNull,
    );

    cache.record('receipt.jpg', Uint8List.fromList([1, 2, 3]), null);
    expect(cache.lookup('receipt.jpg', Uint8List.fromList([1, 2, 3])), isNull);
    cache.record('receipt.jpg', Uint8List.fromList([1, 2, 3]), []);
    expect(cache.lookup('receipt.jpg', Uint8List.fromList([1, 2, 3])), isEmpty);
  });

  test('source word quads use the same exact-byte ownership', () {
    final cache = DetectedTextCache();
    final word = DocumentWordGeometry.fromNative({
      'text': 'GẶP',
      'confidence': .9,
      'points': [
        [.1, .2],
        [.3, .2],
        [.3, .3],
        [.1, .3],
      ],
    })!;
    final words = [word];
    cache.record('receipt.jpg', Uint8List.fromList([1, 2, 3]), [
      'GẶP',
    ], words: words);
    words.clear();
    expect(
      cache
          .lookupWords('receipt.jpg', Uint8List.fromList([1, 2, 3]))
          ?.single
          .text,
      'GẶP',
    );
    expect(
      cache.lookupWords('receipt.jpg', Uint8List.fromList([1, 2, 4])),
      isNull,
    );
    cache.record('receipt.jpg', Uint8List.fromList([1, 2, 3]), null);
    expect(
      cache.lookupWords('receipt.jpg', Uint8List.fromList([1, 2, 3])),
      isNull,
    );
  });
}
