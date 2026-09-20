import 'dart:io';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:tracker_engine/tracker_engine.dart';

import 'flatten_image.dart';
import 'interior_mesh.dart';
import 'document_orientation.dart';
import 'mesh_preview.dart';

class FlattenPreview extends StatefulWidget {
  const FlattenPreview({
    super.key,
    required this.bytes,
    this.measurementBytes,
    required this.boundary,
    required this.name,
    required this.outputDirectory,
    this.textLines = const [],
    this.estimatedRatio,
    this.initialResult,
    this.onResult,
  });
  final Uint8List bytes;
  final Uint8List? measurementBytes;
  final BoundaryResult boundary;
  final String name, outputDirectory;
  final List<String> textLines;
  final double? estimatedRatio;
  final FlattenedImage? initialResult;
  final ValueChanged<FlattenedImage>? onResult;
  @override
  State<FlattenPreview> createState() => FlattenPreviewState();
}

class FlattenPreviewState extends State<FlattenPreview> {
  FlattenedImage? result;
  String? error;
  bool curves = true, busy = true, saving = false;
  InteriorMeshJob? meshJob;
  DocumentOrientationJob? orientationJob;
  @override
  void dispose() {
    meshJob?.cancel();
    orientationJob?.cancel();
    super.dispose();
  }

  int revision = 0;
  @override
  void initState() {
    super.initState();
    if (widget.initialResult != null) {
      result = widget.initialResult;
      busy = false;
    } else {
      _render();
    }
  }

  @override
  void didUpdateWidget(covariant FlattenPreview oldWidget) {
    super.didUpdateWidget(oldWidget);
    // The parent deliberately reuses one keyed preview while navigating.
    // Refresh the state when that key now represents another document; this
    // prevents the previous image from remaining visible and avoids rerendering
    // a completed cached result.
    final oldCorners = oldWidget.boundary.corners ?? const [];
    final newCorners = widget.boundary.corners ?? const [];
    final cornersChanged =
        oldCorners.length != newCorners.length ||
        List.generate(
          oldCorners.length < newCorners.length
              ? oldCorners.length
              : newCorners.length,
          (i) =>
              oldCorners[i].x != newCorners[i].x ||
              oldCorners[i].y != newCorners[i].y,
        ).contains(true);
    final changed =
        oldWidget.name != widget.name ||
        cornersChanged ||
        oldWidget.boundary.width != widget.boundary.width ||
        oldWidget.boundary.height != widget.boundary.height ||
        !listEquals(oldWidget.bytes, widget.bytes) ||
        !listEquals(oldWidget.measurementBytes, widget.measurementBytes) ||
        !listEquals(oldWidget.textLines, widget.textLines) ||
        oldWidget.estimatedRatio != widget.estimatedRatio ||
        jsonEncode(oldWidget.boundary.boundarySegments.map((s) => s.toJson()).toList()) !=
            jsonEncode(widget.boundary.boundarySegments.map((s) => s.toJson()).toList());
    if (!changed) return;
    revision++;
    meshJob?.cancel();
    orientationJob?.cancel();
    error = null;
    saving = false;
    if (widget.initialResult != null) {
      setState(() {
        result = widget.initialResult;
        busy = false;
      });
    } else {
      result = null;
      setState(() => busy = true);
      _render();
    }
  }

  Future<void> _render() async {
    final request = ++revision;
    setState(() {
      busy = true;
      error = null;
      result = null;
    });
    try {
      meshJob?.cancel();
      final preliminary = await flattenImage(
        widget.measurementBytes ?? widget.bytes,
        widget.boundary,
        // Preserve the accepted preliminary correction as the mesh's starting
        // surface; local refinement must not discard the lower-page fix.
        followCurves: curves,
        textLines: widget.textLines,
        estimatedRatio: widget.estimatedRatio,
        maxDimension: const int.fromEnvironment(
          'DOCTRACK_MESH_RESOLUTION',
          defaultValue: 1280,
        ),
        repairBorder: false,
      );
      if (!mounted || revision != request) return;
      final job = InteriorMeshJob();
      meshJob = job;
      final working =
          const int.fromEnvironment(
                'DOCTRACK_MESH_RESOLUTION',
                defaultValue: 1280,
              ) >
              640
          ? await flattenImage(
              widget.measurementBytes ?? widget.bytes,
              widget.boundary,
              followCurves: curves,
              textLines: widget.textLines,
              estimatedRatio: widget.estimatedRatio,
              maxDimension: 640,
              repairBorder: false,
            )
          : null;
      if (!mounted || revision != request) return;
      var mesh = curves
          ? await job.run(preliminary.png, fallbackPng: working?.png)
          : null;
      if (curves && mesh != null && mesh.method.startsWith('Identity mesh')) {
        // Every document gets a second, slower pass before accepting an
        // identity result. This is the advanced mesh stage for folds and
        // weakly printed guides; it never falls back to outline flattening.
        mesh = await job.run(
          preliminary.png,
          fallbackPng: working?.png,
          advanced: true,
        );
      }
      if (curves && mesh == null) {
        throw StateError(
          'Content mesh could not be produced; refusing outline fallback.',
        );
      }
      if (!mounted || revision != request) return;
      final output = await flattenImage(
        widget.bytes,
        widget.boundary,
        followCurves: curves,
        textLines: widget.textLines,
        estimatedRatio: widget.estimatedRatio,
        interiorMesh: mesh,
        // The displayed/saved result must include the separate edge-repair pass.
        // Mesh sampling remains the source of the interior correction; repair
        // only fills the explicit missing fringe.
        repairBorder: true,
      );
      if (!mounted || revision != request) return;
      final orient = DocumentOrientationJob();
      orientationJob = orient;
      final upright = await orient.run(output);
      if (!mounted || revision != request) return;
      setState(() => result = upright);
      widget.onResult?.call(upright);
    } catch (e) {
      if (mounted && revision == request) setState(() => error = e.toString());
    } finally {
      if (mounted && revision == request) setState(() => busy = false);
    }
  }

  Future<void> saveOutput() async {
    final output = result;
    if (output == null || saving) return;
    setState(() => saving = true);
    try {
      final directory = Directory(widget.outputDirectory);
      await directory.create(recursive: true);
      final stem = widget.name
          .replaceAll(RegExp(r'\.[^.]+$'), '')
          .replaceAll(RegExp(r'[^a-zA-Z0-9 _.-]'), '_');
      final file = File(
        '${directory.path}/$stem Flat ${DateTime.now().microsecondsSinceEpoch}.png',
      );
      await file.writeAsBytes(output.png, flush: true);
      if (mounted) {
        ScaffoldMessenger.of(context)
            .showSnackBar(const SnackBar(content: Text('Flattened PNG saved')));
      }
    } catch (e) {
      if (mounted) setState(() => error = 'Could not save: $e');
    } finally {
      if (mounted) setState(() => saving = false);
    }
  }

  @override
  Widget build(BuildContext context) => busy
      ? const Center(child: CircularProgressIndicator(strokeWidth: 2))
      : error != null
      ? Center(child: Text(error!))
      : result == null
      ? const SizedBox()
      : InteractiveViewer(
          minScale: .5,
          maxScale: 6,
          child: MeshPreview(
            bytes: result!.png,
            width: result!.width,
            height: result!.height,
            showMesh: false,
          ),
        );
}
