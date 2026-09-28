import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/scanner/boundary_text_recognizer.dart';
import 'package:pen_tool/flattening/document_words.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('doctrack/boundary-text-test');
  final messenger =
      TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
  tearDown(() => messenger.setMockMethodCallHandler(channel, null));

  test('supported transaction uses one accurate upright request', () async {
    final calls = <String>[];
    const quick = ['TOTAL', '1,234.00', '25.00', '99.50'];
    messenger.setMockMethodCallHandler(channel, (call) async {
      calls.add(call.method);
      return quick;
    });
    final lines = await recognizeBoundaryText(
      Uint8List.fromList([1, 2, 3]),
      id: 4,
      channel: channel,
      fastTransaction: true,
    );
    expect(lines, quick);
    expect(calls, ['recognizeDocumentTextUpright']);
  });

  test(
    'transaction shares its one OCR request with source word geometry',
    () async {
      final calls = <String>[];
      final observed = <DocumentWordGeometry>[];
      messenger.setMockMethodCallHandler(channel, (call) async {
        calls.add(call.method);
        return {
          'lines': ['TOTAL', '1,234.00', '25.00', '99.50'],
          'words': [
            {
              'text': 'TOTAL',
              'confidence': .9,
              'points': [
                [.1, .2],
                [.3, .2],
                [.3, .3],
                [.1, .3],
              ],
            },
            {'text': 'bad', 'confidence': .9, 'points': <Object>[]},
          ],
        };
      });
      final lines = await recognizeBoundaryText(
        Uint8List.fromList([1, 2, 3]),
        id: 7,
        channel: channel,
        fastTransaction: true,
        onWords: observed.addAll,
      );
      expect(lines, ['TOTAL', '1,234.00', '25.00', '99.50']);
      expect(observed.map((word) => word.text), ['TOTAL']);
      expect(calls, ['recognizeDocumentTextUprightWithWords']);
    },
  );

  test(
    'combined identity response discards its words and uses full OCR',
    () async {
      final calls = <String>[];
      final observed = <DocumentWordGeometry>[];
      messenger.setMockMethodCallHandler(channel, (call) async {
        calls.add(call.method);
        if (call.method == 'recognizeDocumentTextUprightWithWords') {
          return {
            'lines': ['IDENTITY CARD', 'NATIONALITY', 'DATE OF BIRTH'],
            'words': [
              {
                'text': 'IDENTITY',
                'confidence': .9,
                'points': [
                  [.1, .2],
                  [.3, .2],
                  [.3, .3],
                  [.1, .3],
                ],
              },
            ],
          };
        }
        return ['PASSPORT', 'NATIONALITY', 'SURNAME'];
      });
      final lines = await recognizeBoundaryText(
        Uint8List.fromList([8]),
        id: 8,
        channel: channel,
        fastTransaction: true,
        onWords: observed.addAll,
      );
      expect(lines, ['PASSPORT', 'NATIONALITY', 'SURNAME']);
      expect(observed, isEmpty);
      expect(calls, [
        'recognizeDocumentTextUprightWithWords',
        'recognizeDocumentText',
      ]);
    },
  );

  test('identity evidence keeps the established full OCR path', () async {
    final calls = <String>[];
    messenger.setMockMethodCallHandler(channel, (call) async {
      calls.add(call.method);
      return call.method == 'recognizeDocumentTextUpright'
          ? ['IDENTITY CARD', 'NATIONALITY', 'DATE OF BIRTH']
          : ['PASSPORT', 'NATIONALITY', 'SURNAME'];
    });
    final lines = await recognizeBoundaryText(
      Uint8List.fromList([8]),
      id: 5,
      channel: channel,
      fastTransaction: true,
    );
    expect(lines, ['PASSPORT', 'NATIONALITY', 'SURNAME']);
    expect(calls, ['recognizeDocumentTextUpright', 'recognizeDocumentText']);
  });

  test('default route calls only the established OCR method', () async {
    final calls = <String>[];
    messenger.setMockMethodCallHandler(channel, (call) async {
      calls.add(call.method);
      return ['unchanged'];
    });
    final lines = await recognizeBoundaryText(
      Uint8List.fromList([9]),
      id: 6,
      channel: channel,
      fastTransaction: false,
    );
    expect(lines, ['unchanged']);
    expect(calls, ['recognizeDocumentText']);
  });
}
