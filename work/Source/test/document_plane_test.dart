import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/document_plane.dart';

void main() {
  for (final viewport in [const Size(536, 560), const Size(1200, 700)]) {
    testWidgets('image and outline share fitted plane in $viewport', (tester) async {
      tester.view.physicalSize = viewport;
      tester.view.devicePixelRatio = 1;
      addTearDown(tester.view.resetPhysicalSize);
      addTearDown(tester.view.resetDevicePixelRatio);
      final side = viewport.shortestSide;
      await tester.pumpWidget(Directionality(
        textDirection: TextDirection.ltr,
        child: InteractiveViewer(
          child: DocumentPlane(
            size: Size.square(side),
            children: [
              Positioned.fill(child: ColoredBox(key: const Key('image'), color: const Color(0xffffffff))),
              Positioned.fill(child: SizedBox(key: const Key('overlay'))),
              Positioned(left: side * .75, top: side * .8,
                child: SizedBox(key: const Key('point'), width: 1, height: 1)),
            ],
          ),
        ),
      ));
      final image = tester.getRect(find.byKey(const Key('image')));
      expect(image.size, Size.square(side));
      expect(image.center, Offset(viewport.width / 2, viewport.height / 2));
      expect(tester.getRect(find.byKey(const Key('overlay'))), image);
      expect(tester.getTopLeft(find.byKey(const Key('point'))),
        image.topLeft + Offset(side * .75, side * .8));
    });
  }
}
