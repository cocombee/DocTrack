import 'dart:convert';
import 'dart:async';

import 'package:tracker_engine/tracker_engine.dart';

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/main.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test(
    'second scan click cancels and stale completion cannot hide reference',
    () async {
      final pending = Completer<BoundaryResult>();
      final started = Completer<void>();
      final store = Store(
        detector: (bytes, cancellation) {
          started.complete();
          return pending.future;
        },
      );
      await store.loadManifest();
      store.points = [PenPoint(x: .2, y: .3)];
      store.dirty = true;
      final scan = store.detectCurrent();
      await started.future;
      store.toggleEngineScan();
      expect(store.detecting, isFalse);
      expect(store.showPrediction, isFalse);
      expect(store.toolMode, isTrue);
      pending.complete(
        BoundaryResult(
          width: 100,
          height: 100,
          status: BoundaryStatus.uncertain,
          outerBoundary: [],
          confidence: .5,
          reasonCodes: [],
          modelVersion: 'test',
          preprocessingVersion: 'test',
        ),
      );
      await scan;
      expect(store.prediction, isNull);
      expect(store.showPrediction, isFalse);
      expect(store.points.single.x, .2);
      expect(store.dirty, isTrue);
      store.dispose();
    },
  );

  test('fresh pixel detection preserves all saved labels and unsaved reference points', () async {
    final store = Store();
    await store.loadManifest();
    final files = Directory('$root/labels')
        .listSync(recursive: true)
        .whereType<File>();
    final before = {for (final f in files) f.path: await f.readAsString()};
    store.points = [PenPoint(x: .01, y: .99, smooth: true)];
    store.dirty = true;
    final reference = jsonEncode(store.points.map((p) => p.toJson()).toList());
    await store.detectCurrent();
    expect(store.detectionError, isNull);
    expect(store.prediction, isNotNull);
    final first = jsonEncode(store.prediction!.toJson());
    store.points = [PenPoint(x: .9, y: .1)];
    await store.detectCurrent();
    expect(jsonEncode(store.prediction!.toJson()), first);
    expect(store.points.single.x, .9);
    expect(store.dirty, isTrue);
    expect(reference, isNotEmpty);
    for (final entry in before.entries) {
      expect(await File(entry.key).readAsString(), entry.value);
    }
    await store.loadImage(1);
    expect(store.showPrediction, isTrue);
    expect(store.prediction, isNotNull);
    final second = store.prediction;
    await store.loadImage(0);
    expect(store.showPrediction, isTrue);
    await store.loadImage(1);
    expect(identical(store.prediction, second), isTrue);
    store.showPrediction = false;
    await store.loadImage(0);
    expect(store.showPrediction, isFalse);
    store.dispose();
  });
}
