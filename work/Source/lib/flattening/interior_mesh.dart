import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:tracker_engine/tracker_engine.dart';

import 'document_words.dart';
import 'mesh_grid_storage.dart';

class InteriorMesh {
  const InteriorMesh(
    this.rows,
    this.columns,
    this.points,
    this.method, {
    this.evidence = const {},
    this.preservedMesh,
    this.refinementRegions = const [],
    this.gridWindow,
    this.coordinateTransform = false,
  });
  final Map<String, dynamic> evidence;
  final int rows, columns;
  final List<TrackerPoint> points;
  final String method;
  final InteriorMesh? preservedMesh;
  final List<List<double>> refinementRegions;

  /// Full grid rows/columns, then this stored window's row/column offsets.
  /// Keeping global integer indices preserves the original interpolation.
  final List<int>? gridWindow;

  /// Apply this local coordinate change through the retained parent map.
  /// Keeping the maps separate avoids folding their sampled approximation.
  final bool coordinateTransform;
  factory InteriorMesh.identity({int rows = 49, int columns = 49}) =>
      InteriorMesh(rows, columns, [
        for (var y = 0; y < rows; y++)
          for (var x = 0; x < columns; x++)
            TrackerPoint(x / (columns - 1), y / (rows - 1)),
      ], 'Identity mesh · insufficient deformation evidence');
  factory InteriorMesh.fromJson(Map<String, dynamic> d) {
    final rows = d['rows'] as int, cols = d['columns'] as int;
    final p = readMeshGrid(d, rows, cols);
    if (rows < 2 ||
        cols < 2 ||
        p.length != rows * cols ||
        p.any((v) => !v.x.isFinite || !v.y.isFinite)) {
      throw const FormatException('Invalid dewarping mesh');
    }
    final coordinateData = d['coordinate_transform'];
    if (coordinateData != null && coordinateData is! bool) {
      throw const FormatException('Invalid coordinate transform');
    }
    final coordinateTransform = coordinateData == true;
    for (var y = 0; y < rows - 1; y++) {
      for (var x = 0; x < cols - 1; x++) {
        final a = p[y * cols + x],
            b = p[y * cols + x + 1],
            c = p[(y + 1) * cols + x];
        if ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x) <= 0) {
          throw const FormatException('Folded dewarping mesh');
        }
        if (coordinateTransform) {
          final bottomRight = p[(y + 1) * cols + x + 1];
          double area(TrackerPoint du, TrackerPoint dv) =>
              du.x * dv.y - du.y * dv.x;
          final top = TrackerPoint(b.x - a.x, b.y - a.y);
          final bottom = TrackerPoint(bottomRight.x - c.x, bottomRight.y - c.y);
          final left = TrackerPoint(c.x - a.x, c.y - a.y);
          final right = TrackerPoint(bottomRight.x - b.x, bottomRight.y - b.y);
          if (area(top, right) <= 0 ||
              area(bottom, left) <= 0 ||
              area(bottom, right) <= 0) {
            throw const FormatException('Folded coordinate transform');
          }
        }
      }
    }
    final preserved = d['preserved_mesh'] == null
        ? null
        : InteriorMesh.fromJson(d['preserved_mesh'] as Map<String, dynamic>);
    if (coordinateTransform && preserved == null) {
      throw const FormatException('Coordinate transform requires its parent');
    }
    final regions = [
      for (final region in (d['refinement_regions'] as List? ?? const []))
        [for (final value in region as List) (value as num).toDouble()],
    ];
    final windowData = d['grid_window'];
    List<int>? window;
    if (windowData != null) {
      if (windowData is! List ||
          windowData.length != 4 ||
          windowData.any((v) => v is! int)) {
        throw const FormatException('Invalid local grid window');
      }
      window = List<int>.unmodifiable(windowData.cast<int>());
      if (preserved == null ||
          window[0] < rows ||
          window[1] < cols ||
          window[2] < 0 ||
          window[3] < 0 ||
          window[2] + rows > window[0] ||
          window[3] + cols > window[1] ||
          regions.any(
            (r) =>
                r.length != 4 ||
                // Compare in the original integer grid coordinates. A region
                // serialized from the same boundary can differ by one ULP
                // from division here, despite addressing exactly that node.
                r[0] * (window![1] - 1) < window[3] ||
                r[1] * (window[0] - 1) < window[2] ||
                r[2] * (window[1] - 1) > window[3] + cols - 1 ||
                r[3] * (window[0] - 1) > window[2] + rows - 1,
          )) {
        throw const FormatException(
          'Local grid does not cover its refinements',
        );
      }
    }
    if ((preserved == null) != regions.isEmpty ||
        regions.any(
          (r) =>
              r.length != 4 ||
              r.any((v) => !v.isFinite || v < 0 || v > 1) ||
              r[0] >= r[2] ||
              r[1] >= r[3],
        )) {
      throw const FormatException('Invalid local refinement regions');
    }
    return InteriorMesh(
      rows,
      cols,
      p,
      d['method'] as String,
      evidence: {...d}
        ..remove('grid')
        ..remove('grid_encoding')
        ..remove('grid_data')
        ..remove('preserved_mesh'),
      preservedMesh: preserved,
      refinementRegions: regions,
      gridWindow: window,
      coordinateTransform: coordinateTransform,
    );
  }
  TrackerPoint at(double u, double v) {
    if (preservedMesh != null &&
        !refinementRegions.any(
          (r) => u >= r[0] && u <= r[2] && v >= r[1] && v <= r[3],
        )) {
      return preservedMesh!.at(u, v);
    }
    final window = gridWindow;
    final x = window == null
        ? (u * (columns - 1)).clamp(0, columns - 1 - 1e-9)
        : (u * (window[1] - 1)).clamp(0, window[1] - 1 - 1e-9) - window[3];
    final y = window == null
        ? (v * (rows - 1)).clamp(0, rows - 1 - 1e-9)
        : (v * (window[0] - 1)).clamp(0, window[0] - 1 - 1e-9) - window[2];
    final ix = x.floor().clamp(0, columns - 2),
        iy = y.floor().clamp(0, rows - 2),
        fx = x - ix,
        fy = y - iy;
    final a = points[iy * columns + ix],
        b = points[iy * columns + ix + 1],
        c = points[(iy + 1) * columns + ix],
        d = points[(iy + 1) * columns + ix + 1];
    double mix(double a, double b, double c, double d) =>
        a * (1 - fx) * (1 - fy) +
        b * fx * (1 - fy) +
        c * (1 - fx) * fy +
        d * fx * fy;
    if (coordinateTransform) {
      final fullColumns = window?[1] ?? columns;
      final fullRows = window?[0] ?? rows;
      final column = ix + (window?[3] ?? 0);
      final row = iy + (window?[2] ?? 0);
      final x0 = column / (fullColumns - 1);
      final x1 = (column + 1) / (fullColumns - 1);
      final y0 = row / (fullRows - 1);
      final y1 = (row + 1) / (fullRows - 1);
      // Interpolate displacement so an unchanged band forwards the exact
      // incoming coordinates, preserving its already working raster pixels.
      return preservedMesh!.at(
        u + mix(a.x - x0, b.x - x1, c.x - x0, d.x - x1),
        v + mix(a.y - y0, b.y - y0, c.y - y1, d.y - y1),
      );
    }
    return TrackerPoint(mix(a.x, b.x, c.x, d.x), mix(a.y, b.y, c.y, d.y));
  }
}

/// Local research-host adapter. No documents are uploaded to any service.
class InteriorMeshJob {
  InteriorMeshJob({
    this.executablePath,
    this.scriptPath,
    this.timeLimit,
    this.wordRecognizer,
  });
  final String? executablePath, scriptPath;
  final Duration? timeLimit;
  final Future<List<DocumentWordGeometry>> Function(Uint8List)? wordRecognizer;
  Process? _process;
  DocumentWordGeometryJob? _wordJob;
  bool _cancelled = false;
  bool _started = false;
  final _evidenceStopped = Completer<void>();
  void cancel() {
    _cancelled = true;
    if (!_evidenceStopped.isCompleted) _evidenceStopped.complete();
    _process?.kill();
    _wordJob?.cancel();
  }

  Future<InteriorMesh> run(
    Uint8List png, {
    Uint8List? fallbackPng,
    bool advanced = false,
  }) async {
    if (_cancelled || _started) return InteriorMesh.identity();
    _started = true;
    const root =
        '/Users/mrkokom/Library/CloudStorage/GoogleDrive-cocombees@gmail.com/My Drive/Codex/Apps/DocTrack/work';
    final python = File(
      executablePath ?? '$root/Scanner Integration/.model-eval/bin/python',
    );
    final script = File(
      scriptPath ??
          const String.fromEnvironment(
            'DOCTRACK_MESH_SCRIPT',
            defaultValue: '$root/Source/tool/protected_ink_mesh.py',
          ),
    );
    if (!await python.exists() || !await script.exists()) {
      return InteriorMesh.identity();
    }
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
      final stdoutDone = _serveWordRequests(process.stdout, temp);
      final stderrDone = process.stderr.drain<void>();
      final exit = await process.exitCode.timeout(
        // Dense local corrections can take minutes on the research host.
        // Allow that measured workload without replacing a safe checkpoint
        // just because the former 45-second preview deadline expired.
        timeLimit ?? Duration(minutes: advanced ? 10 : 5),
        onTimeout: () {
          process.kill();
          return -1;
        },
      );
      if (!_evidenceStopped.isCompleted) _evidenceStopped.complete();
      if (exit != 0) _wordJob?.cancel();
      await Future.wait([stdoutDone, stderrDone]);
      if (_cancelled) return InteriorMesh.identity();
      return await _checkpoint(temp, failed: exit != 0);
    } catch (_) {
      return await _checkpoint(temp, failed: true);
    } finally {
      _process = null;
      if (!_evidenceStopped.isCompleted) _evidenceStopped.complete();
      _wordJob?.cancel();
      _wordJob = null;
      await temp.delete(recursive: true);
    }
  }

  Future<void> _serveWordRequests(
    Stream<List<int>> output,
    Directory folder,
  ) async {
    await for (final line
        in output
            .transform(const Utf8Decoder(allowMalformed: true))
            .transform(const LineSplitter())) {
      if (_cancelled || _evidenceStopped.isCompleted) continue;
      try {
        final message = jsonDecode(line);
        if (message is! Map || message['event'] != 'request_document_words') {
          continue;
        }
        final raster = message['raster'], response = message['response'];
        if (raster is! String || response is! String) continue;
        final name = RegExp(r'^word-evidence-([a-zA-Z0-9-]+)\.png$')
            .firstMatch(raster);
        if (name == null || response != 'word-evidence-${name[1]}.json') {
          continue;
        }
        final image = File('${folder.path}/$raster');
        if (await FileSystemEntity.type(image.path, followLinks: false) !=
            FileSystemEntityType.file) {
          continue;
        }
        final png = await image.readAsBytes();
        if (_cancelled) continue;
        final job = DocumentWordGeometryJob();
        _wordJob = job;
        List<DocumentWordGeometry> words;
        try {
          words =
              await Future.any<List<DocumentWordGeometry>>([
                wordRecognizer == null ? job.run(png) : wordRecognizer!(png),
                _evidenceStopped.future.then((_) => <DocumentWordGeometry>[]),
              ]).timeout(
                const Duration(minutes: 2),
                onTimeout: () {
                  job.cancel();
                  return [];
                },
              );
        } catch (_) {
          words = [];
        }
        if (_cancelled || _evidenceStopped.isCompleted) continue;
        final destination = File('${folder.path}/$response');
        if (await FileSystemEntity.type(destination.path, followLinks: false) !=
            FileSystemEntityType.notFound) {
          continue;
        }
        final temporary = File('${destination.path}.partial');
        if (await FileSystemEntity.type(temporary.path, followLinks: false) !=
            FileSystemEntityType.notFound) {
          continue;
        }
        await temporary.writeAsString(
          jsonEncode(words.map((word) => word.toJson()).toList()),
        );
        await temporary.rename(destination.path);
      } catch (_) {
        // Ordinary helper logs and unavailable optional evidence retain the
        // last accepted mesh; neither may become a new geometry proposal.
      }
    }
  }

  Future<InteriorMesh> _checkpoint(
    Directory temp, {
    required bool failed,
  }) async {
    if (_cancelled) return InteriorMesh.identity();
    try {
      final mesh = InteriorMesh.fromJson(
        jsonDecode(await File('${temp.path}/mesh.json').readAsString())
            as Map<String, dynamic>,
      );
      if (!failed) return mesh;
      return InteriorMesh(
        mesh.rows,
        mesh.columns,
        mesh.points,
        mesh.method,
        evidence: {...mesh.evidence, 'retained_after_process_failure': true},
        preservedMesh: mesh.preservedMesh,
        refinementRegions: mesh.refinementRegions,
        gridWindow: mesh.gridWindow,
        coordinateTransform: mesh.coordinateTransform,
      );
    } catch (_) {
      return InteriorMesh.identity();
    }
  }
}
