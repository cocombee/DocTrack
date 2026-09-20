import 'package:flutter/widgets.dart';

/// Bézier anchor and tangent handles, matched to the 16px toolbar icons.
class CurveRevealIcon extends StatelessWidget {
  const CurveRevealIcon({super.key, required this.color});
  final Color color;
  @override
  Widget build(BuildContext context) => Center(
    child: CustomPaint(size: const Size.square(18), painter: _CurveIconPainter(color)),
  );
}

class _CurveIconPainter extends CustomPainter {
  _CurveIconPainter(this.color);
  final Color color;
  @override
  void paint(Canvas canvas, Size size) {
    canvas.scale(size.width / 24, size.height / 24);
    final pen = Paint()..color = color..style = PaintingStyle.stroke
      ..strokeWidth = 1.8..strokeCap = StrokeCap.round..strokeJoin = StrokeJoin.round;
    canvas.drawLine(const Offset(3, 6), const Offset(9.5, 6), pen);
    canvas.drawLine(const Offset(14.5, 6), const Offset(21, 6), pen);
    canvas.drawPath(Path()..moveTo(5, 16.5)..cubicTo(5, 11, 8, 7, 10, 7), pen);
    canvas.drawPath(Path()..moveTo(14, 7)..cubicTo(16, 7, 19, 11, 19, 16.5), pen);
    for (final rect in [const Rect.fromLTWH(9.5, 3.5, 5, 5),
      const Rect.fromLTWH(2.5, 16.5, 5, 5), const Rect.fromLTWH(16.5, 16.5, 5, 5)]) {
      canvas.drawRRect(RRect.fromRectAndRadius(rect, const Radius.circular(1)), pen);
    }
    for (final p in [const Offset(3, 6), const Offset(21, 6)]) {
      canvas.drawCircle(p, 1.9, Paint()..color = color);
    }
  }
  @override
  bool shouldRepaint(covariant _CurveIconPainter oldDelegate) => oldDelegate.color != color;
}
