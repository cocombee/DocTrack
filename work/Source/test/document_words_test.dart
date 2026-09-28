import 'dart:async';
import 'dart:math';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/flattening/document_words.dart';

Map<String, Object> word() => {
  'text': 'Nguyễn',
  'confidence': .8,
  'points': [
    [.1, .2],
    [.3, .2],
    [.3, .25],
    [.1, .25],
  ],
};

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('doctrack/window');
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  tearDown(() => messenger.setMockMethodCallHandler(channel, null));

  test(
    'word geometry keeps normalized top-left raster coordinates and Unicode',
    () {
      final result = DocumentWordGeometry.fromNative(word())!;
      expect(result.text, 'Nguyễn');
      expect(result.points.first, const Point(.1, .2));
      expect(result.toJson(), word());
      expect(() => result.points.clear(), throwsUnsupportedError);
    },
  );

  test('invalid or uncertain geometry cannot supply word evidence', () {
    for (final invalid in [
      {...word(), 'confidence': .25},
      {...word(), 'confidence': double.nan},
      {...word(), 'text': ''},
      {
        ...word(),
        'points': [
          [1.1, .2],
          [.3, .2],
          [.3, .25],
          [.1, .25],
        ],
      },
      {
        ...word(),
        'points': [
          [.1, .2],
          [.3, .2],
          [.2, .21],
          [.1, .25],
        ],
      },
      {
        ...word(),
        'points': [
          [.1, .2],
          [.1, .2],
          [.1, .2],
          [.1, .2],
        ],
      },
    ]) {
      expect(DocumentWordGeometry.fromNative(invalid), isNull);
    }
  });

  test(
    'native request receives the exact raster and skips malformed observations',
    () async {
      final job = DocumentWordGeometryJob();
      messenger.setMockMethodCallHandler(channel, (call) async {
        expect(call.method, 'recognizeDocumentWords');
        final arguments = call.arguments as Map;
        expect(arguments['id'], job.id);
        expect(arguments['data'], orderedEquals([7, 2, 9]));
        return [
          word(),
          {'points': []},
        ];
      });
      final result = await job.run(Uint8List.fromList([7, 2, 9]));
      expect(result.single.toJson(), word());
    },
  );

  test('cancellation ignores a late native result', () async {
    final pending = Completer<Object?>();
    final started = Completer<void>();
    final job = DocumentWordGeometryJob();
    final cancelled = <int>[];
    messenger.setMockMethodCallHandler(channel, (call) async {
      if (call.method == 'cancelDocumentText') {
        cancelled.add(call.arguments as int);
        return null;
      }
      started.complete();
      return await pending.future;
    });
    final answer = job.run(Uint8List(0));
    await started.future;
    job.cancel();
    pending.complete([word()]);
    expect(await answer, isEmpty);
    expect(cancelled, [job.id]);
  });

  test(
    'cancellation finishes without waiting for a stalled native response',
    () async {
      final pending = Completer<Object?>();
      final started = Completer<void>();
      messenger.setMockMethodCallHandler(channel, (call) async {
        if (call.method == 'cancelDocumentText') return null;
        started.complete();
        return pending.future;
      });
      final job = DocumentWordGeometryJob();
      final answer = job.run(Uint8List(0));
      await started.future;
      job.cancel();
      expect(await answer.timeout(const Duration(seconds: 1)), isEmpty);
      pending.complete([word()]);
    },
  );

  test('an unexpected native container cannot supply word geometry', () async {
    messenger.setMockMethodCallHandler(
      channel,
      (_) async => {'unexpected': word()},
    );
    expect(await DocumentWordGeometryJob().run(Uint8List(0)), isEmpty);
  });

  test(
    'one job cannot reuse its native request id for a second raster',
    () async {
      final pending = Completer<Object?>();
      final started = Completer<void>();
      var calls = 0;
      messenger.setMockMethodCallHandler(channel, (call) async {
        calls++;
        expect((call.arguments as Map)['data'], orderedEquals([1]));
        started.complete();
        return await pending.future;
      });
      final job = DocumentWordGeometryJob();
      final first = job.run(Uint8List.fromList([1]));
      await started.future;
      expect(await job.run(Uint8List.fromList([2])), isEmpty);
      pending.complete([word()]);
      expect((await first).single.text, 'Nguyễn');
      expect(calls, 1);
    },
  );

  test('timeout cancels the native request and supplies no geometry', () async {
    final pending = Completer<Object?>();
    final job = DocumentWordGeometryJob();
    final cancelled = <int>[];
    messenger.setMockMethodCallHandler(channel, (call) async {
      if (call.method == 'cancelDocumentText') {
        cancelled.add(call.arguments as int);
        return null;
      }
      return await pending.future;
    });
    expect(
      await job.run(Uint8List(0), timeout: const Duration(milliseconds: 10)),
      isEmpty,
    );
    pending.complete([word()]);
    await Future<void>.delayed(Duration.zero);
    expect(cancelled, [job.id]);
  });

  test('unsupported native recognition supplies no geometry', () async {
    expect(await DocumentWordGeometryJob().run(Uint8List(0)), isEmpty);
  });
}
