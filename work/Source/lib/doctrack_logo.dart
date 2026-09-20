import 'package:flutter/material.dart';

class DocTrackLogo extends StatelessWidget {
  const DocTrackLogo({super.key, this.size = 32});
  final double size;
  @override
  Widget build(BuildContext context) => Semantics(
    label: 'DocTrack logo',
    child: CustomPaint(size: Size.square(size), painter: _LogoPainter()),
  );
}

class _LogoPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    canvas.scale(size.width / 64, size.height / 64);
    canvas.drawRRect(RRect.fromRectAndRadius(const Rect.fromLTWH(0, 0, 64, 64), const Radius.circular(15)),
      Paint()..color = const Color(0xFF17382D));
    final paper = Path()..moveTo(20, 13)..lineTo(36, 13)..lineTo(45, 22)
      ..lineTo(45, 51)..lineTo(20, 51)..close();
    canvas.drawPath(paper, Paint()..color = const Color(0xFFF4F8F5));
    canvas.drawPath(Path()..moveTo(36, 13)..lineTo(36, 22)..lineTo(45, 22),
      Paint()..color = const Color(0xFF82B99D)..style = PaintingStyle.stroke..strokeWidth = 2);
    final ink = Paint()..color = const Color(0xFF688779)..strokeWidth = 2..strokeCap = StrokeCap.round;
    canvas.drawLine(const Offset(26, 31), const Offset(39, 31), ink);
    canvas.drawLine(const Offset(26, 37), const Offset(36, 37), ink);
    final tracker = Paint()..color = const Color(0xFF62E6A6)..style = PaintingStyle.stroke..strokeWidth = 2;
    canvas.drawRRect(RRect.fromRectAndRadius(const Rect.fromLTWH(15, 9, 35, 46), const Radius.circular(5)), tracker);
    for (final p in [const Offset(15, 14), const Offset(50, 14), const Offset(15, 50), const Offset(50, 50)]) {
      canvas.drawCircle(p, 3, Paint()..color = const Color(0xFF62E6A6));
    }
  }
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
