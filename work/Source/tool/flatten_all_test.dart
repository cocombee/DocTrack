import 'dart:io';
import 'dart:convert';

import 'package:image/image.dart' as img;

import 'package:pen_tool/flattening/isolate_document.dart';
import 'package:pen_tool/flattening/interior_mesh.dart';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:pen_tool/flattening/flatten_image.dart';
import 'package:pen_tool/flattening/flatten_geometry.dart';
import 'package:pen_tool/flattening/rounded_outline.dart';
import 'package:pen_tool/flattening/document_format.dart';

void main() {
  test('flatten accepted detector outputs without changing them', () async {
    final m =
        jsonDecode(File('../Data/manifest.json').readAsStringSync()) as List;
    const predictions = String.fromEnvironment(
      'FLATTEN_PREDICTIONS',
      defaultValue: '../Scanner Integration/Predictions Physical V18',
    );
    const selection = String.fromEnvironment('FLATTEN_SELECTION');
    const output = String.fromEnvironment(
      'FLATTEN_OUTPUT',
      defaultValue: '../Scanner Integration/Flattening Preview V2',
    );
    const ocrPath = String.fromEnvironment('FLATTEN_OCR');
    final ocr = ocrPath.isEmpty
        ? <String, dynamic>{}
        : jsonDecode(File(ocrPath).readAsStringSync()) as Map<String, dynamic>;
    final folder = Directory(output);
    folder.createSync(recursive: true);
    final report = <Map<String, dynamic>>[];
    TrackerPoint p(dynamic x) =>
        TrackerPoint((x['x'] as num).toDouble(), (x['y'] as num).toDouble());
    for (var i = 0; i < m.length; i++) {
      if (selection.isNotEmpty && !selection.split(',').contains('${i + 1}')) {
        continue;
      }
      final e = m[i];
      final d = jsonDecode(
        File('$predictions/${e['name']}.json').readAsStringSync(),
      );
      final boundary = BoundaryResult(
        width: d['width'],
        height: d['height'],
        status: BoundaryStatus.uncertain,
        outerBoundary: [],
        corners: (d['corners'] as List).map(p).toList(),
        boundarySegments: [
          for (final s in d['boundarySegments'])
            BoundarySegment(
              p(s['start']),
              p(s['end']),
              control1: s['control1'] == null ? null : p(s['control1']),
              control2: s['control2'] == null ? null : p(s['control2']),
              isImageClip: s['isImageClip'] == true,
            ),
        ],
        confidence: .5,
        modelVersion: 'accepted41',
        preprocessingVersion: 'upright',
      );
      try {
        final textLines = List<String>.from(ocr[e['name']] ?? const []);
        if (const bool.fromEnvironment('FLATTEN_GEOMETRY_ONLY')) {
          final geometry = FlattenGeometry(
            boundary.corners!,
            boundary.boundarySegments,
            boundary.width,
            boundary.height,
          );
          final shape = RoundedOutline(geometry, 640, 640);
          report.add({
            'index': i + 1,
            'roundedCorners': shape.corners.toList(),
            'radius': shape.radius,
            'ratio': geometry.estimatedRatio,
            'format': identifyDocumentFormat(
              geometry.estimatedRatio,
              textLines,
            ).label,
            'content': DocumentContentEvidence.fromLines(textLines).kind.name,
          });
          continue;
        }
        const sourceOverride = String.fromEnvironment(
          'FLATTEN_SOURCE_OVERRIDE',
        );
        final candidateSource = File('$sourceOverride/${e['name']}.png');
        final sourceFile =
            sourceOverride.isNotEmpty && candidateSource.existsSync()
            ? candidateSource
            : File('../Data/${e['image']}');
        if (const bool.fromEnvironment('FLATTEN_EXPORT_SOURCE')) {
          final decoded = img.bakeOrientation(
            img.decodeImage(await sourceFile.readAsBytes())!,
          );
          await File('${folder.path}/${e['name']}.png')
              .writeAsBytes(img.encodePng(decoded));
          report.add({
            'index': i + 1,
            'width': decoded.width,
            'height': decoded.height,
          });
          continue;
        }
        final selected = await isolateDocument(
          await sourceFile.readAsBytes(),
          boundary,
          textLines: textLines,
        );
        if (const bool.fromEnvironment('FLATTEN_ISOLATE_ONLY')) {
          await File('${folder.path}/Image ${i + 1} Isolated.png')
              .writeAsBytes(selected.png);
          report.add({
            'index': i + 1,
            'estimatedRatio': selected.estimatedRatio,
          });
          continue;
        }
        final measurementPng =
            const bool.fromEnvironment(
              'FLATTEN_STABLE_MEASUREMENT',
              defaultValue: true,
            )
            ? selected.measurementPng ?? selected.png
            : selected.png;
        final preliminary = await flattenImage(
          measurementPng,
          selected.boundary,
          followCurves: const bool.fromEnvironment(
            'FLATTEN_FOLLOW_CURVES',
            defaultValue: true,
          ),
          maxDimension: const int.fromEnvironment(
            'MESH_RESOLUTION',
            defaultValue: 1280,
          ),
          textLines: textLines,
          estimatedRatio: selected.estimatedRatio,
          repairBorder: false,
        );
        await File('${folder.path}/Image ${i + 1} Preliminary.png')
            .writeAsBytes(preliminary.png);
        final working =
            const int.fromEnvironment('MESH_RESOLUTION', defaultValue: 1280) >
                640
            ? await flattenImage(
                measurementPng,
                selected.boundary,
                followCurves: const bool.fromEnvironment(
                  'FLATTEN_FOLLOW_CURVES',
                  defaultValue: true,
                ),
                maxDimension: 640,
                textLines: textLines,
                estimatedRatio: selected.estimatedRatio,
                repairBorder: false,
              )
            : null;
        final mesh = await InteriorMeshJob().run(
          preliminary.png,
          fallbackPng: working?.png,
        );
        if (const bool.fromEnvironment('FLATTEN_REQUIRE_MESH') &&
            mesh == null) {
          throw StateError(
            'Mesh engine failed; outline fallback is not validation',
          );
        }
        if (selection.isNotEmpty) {
          final raw = await flattenImage(
            selected.png,
            selected.boundary,
            followCurves: const bool.fromEnvironment(
              'FLATTEN_FOLLOW_CURVES',
              defaultValue: true,
            ),
            maxDimension: const int.fromEnvironment(
              'FLATTEN_OUTPUT_RESOLUTION',
              defaultValue: 640,
            ),
            textLines: textLines,
            estimatedRatio: selected.estimatedRatio,
            interiorMesh: mesh,
            repairBorder: false,
          );
          await File('${folder.path}/Image ${i + 1} Raw.png')
              .writeAsBytes(raw.png);
        }
        final out = await flattenImage(
          selected.png,
          selected.boundary,
          followCurves: const bool.fromEnvironment(
            'FLATTEN_FOLLOW_CURVES',
            defaultValue: true,
          ),
          maxDimension: const int.fromEnvironment(
            'FLATTEN_OUTPUT_RESOLUTION',
            defaultValue: 640,
          ),
          textLines: textLines,
          estimatedRatio: selected.estimatedRatio,
          interiorMesh: mesh,
        );
        stdout.writeln(
          'Validated image ${i + 1}: ${mesh?.method ?? "outline"}',
        );
        await File('${folder.path}/Image ${i + 1}.png').writeAsBytes(out.png);
        report.add({
          'index': i + 1,
          'name': e['name'],
          'width': out.width,
          'height': out.height,
          'format': out.format,
          'mesh': mesh?.method,
          'meshEvidence': mesh?.evidence,
        });
      } catch (error) {
        report.add({'index': i + 1, 'error': error.toString()});
      }
    }
    File('${folder.path}/Report.json').writeAsStringSync(jsonEncode(report));
    expect(report.where((r) => r.containsKey('error')).toList(), isEmpty);
  }, timeout: const Timeout(Duration(minutes: 8)));
}
