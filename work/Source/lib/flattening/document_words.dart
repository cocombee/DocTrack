import 'dart:async';
import 'dart:math';

import 'package:flutter/services.dart';

/// Word quadrilaterals in the exact supplied raster's normalized coordinates.
class DocumentWordGeometry {
  DocumentWordGeometry._(this.text, this.confidence, this.points);
  final String text;
  final double confidence;
  final List<Point<double>> points;

  static DocumentWordGeometry? fromNative(Object? value) {
    if (value is! Map) return null;
    final text = value['text'];
    final confidence = value['confidence'];
    final corners = value['points'];
    if (text is! String ||
        text.trim().isEmpty ||
        confidence is! num ||
        !confidence.isFinite ||
        confidence <= .25 ||
        confidence > 1 ||
        corners is! List ||
        corners.length != 4) {
      return null;
    }
    final points = <Point<double>>[];
    for (final corner in corners) {
      if (corner is! List || corner.length != 2) return null;
      final x = corner[0], y = corner[1];
      if (x is! num ||
          y is! num ||
          !x.isFinite ||
          !y.isFinite ||
          x < 0 ||
          x > 1 ||
          y < 0 ||
          y > 1) {
        return null;
      }
      points.add(Point(x.toDouble(), y.toDouble()));
    }
    final area = List.generate(4, (i) {
      final a = points[i], b = points[(i + 1) % 4];
      return a.x * b.y - a.y * b.x;
    }).reduce((a, b) => a + b);
    if (area.abs() < 1e-12) return null;
    for (var i = 0; i < 4; i++) {
      final a = points[i], b = points[(i + 1) % 4], c = points[(i + 2) % 4];
      final turn = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
      if (turn * area <= 0) return null;
    }
    return DocumentWordGeometry._(
      text,
      confidence.toDouble(),
      List.unmodifiable(points),
    );
  }

  Map<String, Object> toJson() => {
    'text': text,
    'confidence': confidence,
    'points': points.map((p) => [p.x, p.y]).toList(),
  };
}

/// Optional local evidence; unavailable or cancelled recognition supplies none.
class DocumentWordGeometryJob {
  static int _nextId = 2000000;
  static const _channel = MethodChannel('doctrack/window');
  final int id = ++_nextId;
  bool _cancelled = false;
  bool _started = false;
  final _cancelSignal = Completer<Object?>();

  void cancel() {
    _cancelled = true;
    if (!_cancelSignal.isCompleted) _cancelSignal.complete(null);
    _channel
        .invokeMethod<void>('cancelDocumentText', id)
        .catchError((Object _) {});
  }

  Future<List<DocumentWordGeometry>> run(
    Uint8List png, {
    Duration timeout = const Duration(minutes: 2),
  }) async {
    if (_cancelled || _started) return [];
    _started = true;
    try {
      final response =
          await Future.any<Object?>([
            _channel.invokeMethod<Object?>('recognizeDocumentWords', {
              'data': png,
              'id': id,
            }),
            _cancelSignal.future,
          ]).timeout(
            timeout,
            onTimeout: () {
              cancel();
              return null;
            },
          );
      if (_cancelled) return [];
      if (response is! List) return [];
      return response
          .map(DocumentWordGeometry.fromNative)
          .whereType<DocumentWordGeometry>()
          .toList();
    } on MissingPluginException {
      return [];
    } on PlatformException {
      return [];
    }
  }
}
