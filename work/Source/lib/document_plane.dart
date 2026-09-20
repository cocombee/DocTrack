import 'package:flutter/widgets.dart';

/// Keeps the image, overlay and hit targets on one source-proportional plane.
/// Center loosens the viewport's tight constraints before applying [size].
class DocumentPlane extends StatelessWidget {
  const DocumentPlane({super.key, required this.size, required this.children});

  final Size size;
  final List<Widget> children;

  @override
  Widget build(BuildContext context) => Center(
    child: SizedBox(
      width: size.width,
      height: size.height,
      child: Stack(children: children),
    ),
  );
}
