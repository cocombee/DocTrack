import 'package:flutter/foundation.dart';

import '../flattening/document_words.dart';

/// Reuses detection OCR only for the exact photo bytes it described.
class DetectedTextCache {
  final Map<String, _DetectedText> _entries = {};

  void record(
    String imagePath,
    Uint8List bytes,
    List<String>? lines, {
    List<DocumentWordGeometry>? words,
  }) {
    if (lines == null) {
      _entries.remove(imagePath);
      return;
    }
    _entries[imagePath] = _DetectedText(
      Uint8List.fromList(bytes),
      List<String>.unmodifiable(lines),
      words == null ? null : List<DocumentWordGeometry>.unmodifiable(words),
    );
  }

  List<String>? lookup(String imagePath, Uint8List bytes) {
    final entry = _entries[imagePath];
    return entry != null && listEquals(entry.bytes, bytes) ? entry.lines : null;
  }

  List<DocumentWordGeometry>? lookupWords(String imagePath, Uint8List bytes) {
    final entry = _entries[imagePath];
    return entry != null && listEquals(entry.bytes, bytes) ? entry.words : null;
  }
}

class _DetectedText {
  const _DetectedText(this.bytes, this.lines, this.words);
  final Uint8List bytes;
  final List<String> lines;
  final List<DocumentWordGeometry>? words;
}
