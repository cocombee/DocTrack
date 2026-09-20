import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'flatten_geometry.dart';
import 'interior_mesh.dart';

class MeshPreview extends StatelessWidget {
  const MeshPreview({
    super.key,
    required this.bytes,
    required this.width,
    required this.height,
    required this.showMesh,
    this.geometry,
    this.interior,
  });
  final Uint8List bytes;
  final int width, height;
  final bool showMesh;
  final FlattenGeometry? geometry;
  final InteriorMesh? interior;
  @override
  Widget build(BuildContext context) => LayoutBuilder(
    builder: (context, box) {
      final size = applyBoxFit(
        BoxFit.contain,
        Size(width.toDouble(), height.toDouble()),
        box.biggest,
      ).destination;
      return Center(
        child: SizedBox(
          width: size.width,
          height: size.height,
          child: Stack(
            fit: StackFit.expand,
            children: [
              Image.memory(bytes, fit: BoxFit.fill),
              if (showMesh)
                IgnorePointer(
                  child: CustomPaint(painter: _MeshPainter(geometry, interior)),
                ),
            ],
          ),
        ),
      );
    },
  );
}

class _MeshPainter extends CustomPainter {
  const _MeshPainter(this.geometry, this.interior);
  final FlattenGeometry? geometry;
  final InteriorMesh? interior;
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0x8838BDF8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = .7;
    for (var axis = 0; axis < 2; axis++) {
      for (var line = 0; line <= 12; line++) {
        final path = Path();
        for (var j = 0; j <= 64; j++) {
          var u = axis == 0 ? line / 12 : j / 64,
              v = axis == 0 ? j / 64 : line / 12;
          if (geometry != null) {
            final q = interior?.at(u, v);
            if (q != null) {
              u = q.x;
              v = q.y;
            }
            final p = geometry!.source(u, v);
            u = p.x;
            v = p.y;
          }
          if (j == 0) {
            path.moveTo(u * size.width, v * size.height);
          } else {
            path.lineTo(u * size.width, v * size.height);
          }
        }
        canvas.drawPath(path, paint);
      }
    }
  }

  @override
  bool shouldRepaint(_MeshPainter old) =>
      old.geometry != geometry || old.interior != interior;
}
