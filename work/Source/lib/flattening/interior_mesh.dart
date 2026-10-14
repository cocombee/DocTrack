import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:tracker_engine/tracker_engine.dart';

class InteriorMesh {
  const InteriorMesh(
    this.rows,
    this.columns,
    this.points,
    this.method, {
    this.evidence = const {},
  });
  final Map<String, dynamic> evidence;
  final int rows, columns;
  final List<TrackerPoint> points;
  final String method;
  factory InteriorMesh.identity({int rows = 49, int columns = 49}) =>
      InteriorMesh(
        rows,
        columns,
        [
          for (var y = 0; y < rows; y++)
            for (var x = 0; x < columns; x++)
              TrackerPoint(x / (columns - 1), y / (rows - 1)),
        ],
        'Identity mesh · insufficient deformation evidence',
      );
  factory InteriorMesh.fromJson(Map<String, dynamic> d) {
    final rows = d['rows'] as int, cols = d['columns'] as int;
    final p = [
      for (final v in d['grid'])
        TrackerPoint((v[0] as num).toDouble(), (v[1] as num).toDouble()),
    ];
    if (rows < 2 ||
        cols < 2 ||
        p.length != rows * cols ||
        p.any((v) => !v.x.isFinite || !v.y.isFinite)) {
      throw const FormatException('Invalid dewarping mesh');
    }
    for (var y = 0; y < rows - 1; y++) {
      for (var x = 0; x < cols - 1; x++) {
        final a = p[y * cols + x],
            b = p[y * cols + x + 1],
            c = p[(y + 1) * cols + x];
        if ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) <= 0) {
          throw const FormatException('Folded dewarping mesh');
        }
      }
    }
    return InteriorMesh(
      rows,
      cols,
      p,
      d['method'] as String,
      evidence: {...d}..remove('grid'),
    );
  }
  TrackerPoint at(double u, double v) {
    final x = (u * (columns - 1)).clamp(0, columns - 1 - 1e-9),
        y = (v * (rows - 1)).clamp(0, rows - 1 - 1e-9);
    final ix = x.floor(), iy = y.floor(), fx = x - ix, fy = y - iy;
    final a = points[iy * columns + ix],
        b = points[iy * columns + ix + 1],
        c = points[(iy + 1) * columns + ix],
        d = points[(iy + 1) * columns + ix + 1];
    double mix(double a, double b, double c, double d) =>
        a * (1 - fx) * (1 - fy) +
        b * fx * (1 - fy) +
        c * (1 - fx) * fy +
        d * fx * fy;
    return TrackerPoint(mix(a.x, b.x, c.x, d.x), mix(a.y, b.y, c.y, d.y));
  }
}

/// Local research-host adapter. No documents are uploaded to any service.
class InteriorMeshJob {
  Process? _process;
  bool _cancelled = false;
  void cancel() {
    _cancelled = true;
    _process?.kill();
  }

  Future<InteriorMesh> run(
    Uint8List png, {
    Uint8List? fallbackPng,
    bool advanced = false,
  }) async {
    const root =
        '/Users/mrkokom/Library/CloudStorage/GoogleDrive-cocombees@gmail.com/My Drive/Codex/Apps/DocTrack/work';
    final python = File('$root/Scanner Integration/.model-eval/bin/python');
    final script = File(
      const String.fromEnvironment(
        'DOCTRACK_MESH_SCRIPT',
        defaultValue: '$root/Source/tool/mesh_frame/dewarp_mesh.py',
      ),
    );
    if (!await python.exists() || !await script.exists()) return InteriorMesh.identity();
    final temp = await Directory.systemTemp.createTemp('doctrack-mesh-');
    try {
      await File('${temp.path}/input.png').writeAsBytes(png);
      if (fallbackPng != null) {
        await File('${temp.path}/working.png').writeAsBytes(fallbackPng);
      }
      if (_cancelled) return InteriorMesh.identity();
      final process = await Process.start(python.path, [
        script.path,
        '${temp.path}/input.png',
        '${temp.path}/mesh.json',
        if (fallbackPng != null) '${temp.path}/working.png',
      ]);
      _process = process;
      if (_cancelled) process.kill();
      final stdoutDone = process.stdout.drain<void>();
      final stderrDone = process.stderr.drain<void>();
      final exit = await process.exitCode.timeout(
        Duration(seconds: advanced ? 120 : 45),
        onTimeout: () {
          process.kill();
          return -1;
        },
      );
      await Future.wait([stdoutDone, stderrDone]);
      if (_cancelled) return InteriorMesh.identity();
      if (exit != 0) return InteriorMesh.identity();
      return InteriorMesh.fromJson(
        jsonDecode(await File('${temp.path}/mesh.json').readAsString())
            as Map<String, dynamic>,
      );
    } catch (_) {
      return InteriorMesh.identity();
    } finally {
      _process = null;
      await temp.delete(recursive: true);
    }
  }
}
