import 'dart:async';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/flattening/document_words.dart';
import 'package:pen_tool/flattening/interior_mesh.dart';

const checkpoint = '''
{"rows":2,"columns":2,"method":"Accepted checkpoint",
"grid":[[0,0],[1,0],[0,1],[1,1]]}
''';

void main() {
  Future<InteriorMesh> run(
    String body, {
    Duration? limit,
    Future<List<DocumentWordGeometry>> Function(Uint8List)? words,
    void Function(InteriorMeshJob)? onJob,
  }) async {
    final folder = await Directory.systemTemp.createTemp('mesh-job-test-');
    try {
      final script = File('${folder.path}/helper.sh');
      await script.writeAsString(body);
      final job = InteriorMeshJob(
        executablePath: '/bin/sh',
        scriptPath: script.path,
        timeLimit: limit,
        wordRecognizer: words,
      );
      onJob?.call(job);
      return await job.run(Uint8List.fromList([7, 19, 31]));
    } finally {
      await folder.delete(recursive: true);
    }
  }

  const writeAccepted =
      '''
cat > "\$2" <<'JSON'
$checkpoint
JSON
''';

  const requestWords = r'''
folder=$(dirname "$2")
cp "$1" "$folder/word-evidence-test.png"
echo '{"event":"request_document_words","raster":"word-evidence-test.png","response":"word-evidence-test.json"}'
i=0
while [ ! -f "$folder/word-evidence-test.json" ]; do
  i=$((i + 1))
  [ "$i" -lt 400 ] || exit 9
  sleep .01
done
''';

  test(
    'helper word evidence uses its exact raster and atomic response',
    () async {
      var calls = 0;
      final result = await run(
        '$writeAccepted$requestWords'
        r'''
case $(cat "$folder/word-evidence-test.json") in
  *'"text":"Invoice"'*) ;;
  *) exit 8 ;;
esac
[ ! -e "$folder/word-evidence-test.json.partial" ] || exit 8
''',
        words: (png) async {
          calls++;
          expect(png, orderedEquals([7, 19, 31]));
          return [
            DocumentWordGeometry.fromNative({
              'text': 'Invoice',
              'confidence': .9,
              'points': [
                [.1, .1],
                [.4, .1],
                [.4, .2],
                [.1, .2],
              ],
            })!,
          ];
        },
      );
      expect(calls, 1);
      expect(result.method, 'Accepted checkpoint');
      expect(result.evidence['retained_after_process_failure'], isNull);
    },
  );

  test('unavailable word evidence preserves the accepted checkpoint', () async {
    final result = await run(
      '$writeAccepted$requestWords'
      r'''[ "$(cat "$folder/word-evidence-test.json")" = '[]' ] || exit 8
''',
      words: (_) async => throw StateError('Unavailable'),
    );
    expect(result.method, 'Accepted checkpoint');
    expect(result.evidence['retained_after_process_failure'], isNull);
  });

  test('unsafe or symlink word requests never reach recognition', () async {
    var calls = 0;
    final result = await run(
      '$writeAccepted'
      r'''
folder=$(dirname "$2")
ln -s "$1" "$folder/word-evidence-link.png"
echo '{"event":"request_document_words","raster":"../input.png","response":"../result.json"}'
echo '{"event":"request_document_words","raster":"word-evidence-link.png","response":"word-evidence-link.json"}'
'''
      '$requestWords',
      words: (_) async {
        calls++;
        return [];
      },
    );
    expect(calls, 1);
    expect(result.evidence['retained_after_process_failure'], isNull);
  });

  test('cancellation does not wait for stalled word recognition', () async {
    final started = Completer<void>();
    final pending = Completer<List<DocumentWordGeometry>>();
    late InteriorMeshJob job;
    final result = run(
      '$writeAccepted$requestWords',
      onJob: (value) => job = value,
      words: (_) {
        started.complete();
        return pending.future;
      },
    );
    await started.future.timeout(const Duration(seconds: 2));
    job.cancel();
    expect(
      (await result.timeout(const Duration(seconds: 1))).method,
      startsWith('Identity mesh'),
    );
    pending.complete([]);
  });

  test(
    'helper exit releases a stalled recognizer and keeps its checkpoint',
    () async {
      final started = Completer<void>();
      final pending = Completer<List<DocumentWordGeometry>>();
      final result = run(
        '$writeAccepted'
        r'''
folder=$(dirname "$2")
cp "$1" "$folder/word-evidence-test.png"
echo '{"event":"request_document_words","raster":"word-evidence-test.png","response":"word-evidence-test.json"}'
sleep .2
exit 7
''',
        words: (_) {
          started.complete();
          return pending.future;
        },
      );
      await started.future.timeout(const Duration(seconds: 2));
      final retained = await result.timeout(const Duration(seconds: 1));
      expect(retained.method, 'Accepted checkpoint');
      expect(retained.evidence['retained_after_process_failure'], isTrue);
      pending.complete([]);
    },
  );

  test(
    'later process failure retains its validated accepted checkpoint',
    () async {
      final result = await run('${writeAccepted}exit 7\n');
      expect(result.method, 'Accepted checkpoint');
      expect(result.evidence['retained_after_process_failure'], isTrue);
      expect(result.at(.3, .7).x, .3);
      expect(result.at(.3, .7).y, .7);
    },
  );

  test('optional step timeout retains the checkpoint before cleanup', () async {
    final result = await run(
      '${writeAccepted}while :; do :; done\n',
      limit: const Duration(seconds: 1),
    );
    expect(result.method, 'Accepted checkpoint');
    expect(result.evidence['retained_after_process_failure'], isTrue);
  });

  test(
    'failed helper retains the coordinate map through its actual parent',
    () async {
      const layered = r'''
cat > "$2" <<'JSON'
{"rows":2,"columns":2,"method":"Accepted coordinate checkpoint",
"grid":[[0,0],[1,0],[0,0.9],[1,0.9]],"coordinate_transform":true,
"refinement_regions":[[0,0,1,1]],
"preserved_mesh":{"rows":2,"columns":2,"method":"Accepted parent",
"grid":[[0,0],[0.8,0],[0,1],[0.8,1]]}}
JSON
''';
      final completed = await run(layered);
      final retained = await run('${layered}exit 7\n');
      expect(retained.coordinateTransform, isTrue);
      expect(retained.evidence['retained_after_process_failure'], isTrue);
      for (final point in [
        [.3, .7],
        [.6, .4],
      ]) {
        final before = completed.at(point[0], point[1]);
        final after = retained.at(point[0], point[1]);
        expect(after.x, before.x);
        expect(after.y, before.y);
      }
      expect(retained.at(.3, .7).x, closeTo(.24, 1e-12));
      expect(retained.at(.3, .7).y, closeTo(.63, 1e-12));
    },
  );

  test('partial invalid output cannot become an accepted checkpoint', () async {
    final result = await run('echo "{partial" > "\$2"\nexit 7\n');
    expect(result.method, startsWith('Identity mesh'));
    expect(result.evidence['retained_after_process_failure'], isNull);
  });
}
