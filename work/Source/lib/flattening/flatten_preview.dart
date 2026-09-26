import 'dart:io';
import 'dart:convert';
import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:tracker_engine/tracker_engine.dart';

import 'flatten_image.dart';
import 'interior_mesh.dart';
import 'document_orientation.dart';
import 'mesh_preview.dart';
import 'document_format.dart';
import 'rigid_document.dart';

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
        // Keep rounded geometry for the outline mask, but do not use long
        // boundary curves as side deformation in the final coordinate map.
        followCurves: false,
        textLines: widget.textLines,
        estimatedRatio: widget.estimatedRatio,
        maxDimension: const int.fromEnvironment(
          'DOCTRACK_MESH_RESOLUTION',
          defaultValue: 1280,
        ),
        repairBorder: false,
      );
      if (!mounted || revision != request) return;
      final edgeRisk = widget.boundary.corners?.any((p) =>
              p.x <= .02 || p.x >= .98 || p.y <= .02 || p.y >= .98) ??
          false;
      final observedRatio = widget.estimatedRatio ??
          widget.boundary.width / widget.boundary.height;
      final format = identifyDocumentFormat(observedRatio, widget.textLines);
      final idCard = format.label.startsWith('ID /');
      final rigidCard = isRigidDocumentForFlattening(
        widget.boundary,
        format,
        widget.textLines,
      );
      double distance(TrackerPoint a, TrackerPoint b) =>
          ((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
      bool nearStraight(BoundarySegment s) {
        if (!s.isCurve) return true;
        final length = math.sqrt(distance(s.start, s.end));
        if (length < 1e-6) return false;
        var maximum = 0.0;
        for (var i = 1; i < 8; i++) {
          final p = s.at(i / 8), a = s.start, b = s.end;
          final dx = b.x - a.x, dy = b.y - a.y;
          final t = ((p.x - a.x) * dx + (p.y - a.y) * dy) /
              (dx * dx + dy * dy);
          final q = TrackerPoint(a.x + t.clamp(0, 1) * dx,
              a.y + t.clamp(0, 1) * dy);
          maximum = math.max(maximum, math.sqrt(distance(p, q)));
        }
        return maximum / length <= .02;
      }
      final flatFourSide = widget.boundary.boundarySegments.length <= 6 &&
          widget.boundary.boundarySegments.every(nearStraight);
      final roundedEvidence = widget.boundary.boundarySegments
          .where((s) => s.isCurve)
          .length;
      final twoPointEvidence = widget.boundary.boundarySegments.where((s) {
        if (s.isCurve || s.isImageClip) return false;
        final dx = (s.end.x - s.start.x) * widget.boundary.width;
        final dy = (s.end.y - s.start.y) * widget.boundary.height;
        return math.sqrt(dx * dx + dy * dy) <=
            math.min(widget.boundary.width, widget.boundary.height) * .2;
      }).length;
      final cornerEvidence = roundedEvidence >= 2 || twoPointEvidence >= 2;
      final standardCard = idCard || format.label.startsWith('Passport');
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
      // Always run the basic mesh once when curves are present so the preview
      // can confirm whether interior correction is actually needed. Rigid IDs
      // and near-straight paper still discard that candidate for final output.
      final diagnosticMesh = curves
          ? await job.run(preliminary.png, fallbackPng: working?.png)
          : null;
      // A straight outer crop does not prove the printed content is flat:
      // receipts can have bowed rows while all four paper edges remain nearly
      // straight. Only rigid ID/passport families suppress mesh application.
      var mesh = rigidCard ? null : diagnosticMesh;
      // Identity is a valid result: perspective-only flattening is the safe
      // path for flat IDs and paper. Do not force an advanced mesh pass after
      // the evidence-gated basic solver declines interior deformation.
      if (curves && mesh == null && !rigidCard) {
        throw StateError(
          'Content mesh could not be produced; refusing outline fallback.',
        );
      }
      if (!mounted || revision != request) return;
      final output = await flattenImage(
        widget.bytes,
        widget.boundary,
        // Use perspective plus the accepted interior mesh. Boundary curves
        // remain available to the rounded outline mask, but do not bow the
        // long sides of otherwise flat documents.
        followCurves: false,
        textLines: widget.textLines,
        estimatedRatio: widget.estimatedRatio,
        interiorMesh: mesh,
        // ID cards get a small protective overscan to remove uncertain edge
        // pixels after perspective fitting. Paper and passports keep their
        // measured crop and are never enlarged by this rule.
        documentScale: standardCard ? 1.03 : 1.0,
        // The displayed/saved result must include the separate edge-repair pass.
        // Mesh sampling remains the source of the interior correction; repair
        // only fills the explicit missing fringe.
        // Perspective-only results keep their measured crop. Edge repair is
        // reserved for a real accepted interior mesh; applying it to an
        // identity result can fill and push the ends inward on flat IDs.
        // Every family gets the explicit fringe repair. Rounded ID outlines
        // constrain that repair to the card silhouette, while rectangular
        // paper uses the measured edge and stock surface.
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
