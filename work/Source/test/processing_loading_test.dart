import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/flattening/processing_loading.dart';

void main() {
  testWidgets(
    'loading keeps elapsed time through stages and closes the preview',
    (tester) async {
      var closed = false;
      final start = DateTime.now().subtract(const Duration(seconds: 20));
      Future<void> show(String stage) => tester.pumpWidget(
        MaterialApp(
          theme: ThemeData.dark(useMaterial3: true),
          home: Scaffold(
            body: ImageProcessingLoading(
              startedAt: start,
              stage: stage,
              onClose: () => closed = true,
            ),
          ),
        ),
      );
      await show('Finding the document…');
      expect(
        find.text('This image is taking longer to straighten'),
        findsOneWidget,
      );
      expect(find.textContaining('s elapsed'), findsOneWidget);
      await show('Straightening text and lines…');
      expect(find.text('Analyzing image…'), findsNothing);
      expect(find.text('Straightening text and lines…'), findsOneWidget);
      await tester.tap(find.byTooltip('Close preview'));
      expect(closed, isTrue);
      await tester.pumpWidget(const SizedBox());
      await tester.pump(const Duration(seconds: 2));
      expect(tester.takeException(), isNull);
    },
  );

  testWidgets('loading fits wide, narrow and enlarged-text preview surfaces', (
    tester,
  ) async {
    final capture = Platform.environment['DOCTRACK_LOADING_REVIEW'];
    if (capture != null) {
      // Flutter's test font paints rectangles, which cannot verify typography.
      // Load the SDK's existing fonts only for the optional review renders.
      final sdk = Platform.environment['FLUTTER_ROOT']!;
      await tester.runAsync(() async {
        for (final font in [
          (family: 'Roboto', file: 'Roboto-Regular.ttf'),
          (family: 'MaterialIcons', file: 'MaterialIcons-Regular.otf'),
        ]) {
          final bytes = await File(
            '$sdk/bin/cache/artifacts/material_fonts/${font.file}',
          ).readAsBytes();
          await (FontLoader(
            font.family,
          )..addFont(Future.value(ByteData.sublistView(bytes)))).load();
        }
      });
    }
    for (final state in [
      (
        name: 'Initial Wide',
        size: const Size(800, 500),
        scale: 1.0,
        delayed: false,
      ),
      (
        name: 'Long Narrow',
        size: const Size(320, 400),
        scale: 1.0,
        delayed: true,
      ),
      (
        name: 'Long Large Text',
        size: const Size(320, 400),
        scale: 2.0,
        delayed: true,
      ),
    ]) {
      tester.view.devicePixelRatio = 1;
      tester.view.physicalSize = state.size;
      final boundary = GlobalKey();
      await tester.pumpWidget(
        MaterialApp(
          theme: ThemeData.dark(useMaterial3: true),
          home: RepaintBoundary(
            key: boundary,
            child: Scaffold(
              backgroundColor: Colors.black,
              body: MediaQuery(
                data: MediaQueryData(
                  size: state.size,
                  textScaler: TextScaler.linear(state.scale),
                ),
                child: ImageProcessingLoading(
                  startedAt: DateTime.now().subtract(
                    Duration(seconds: state.delayed ? 20 : 0),
                  ),
                  stage: 'Straightening text and lines…',
                  onClose: () {},
                ),
              ),
            ),
          ),
        ),
      );
      await tester.pump(const Duration(milliseconds: 250));
      expect(tester.takeException(), isNull);
      expect(
        find.text(
          state.delayed
              ? 'This image is taking longer to straighten'
              : 'Analyzing image…',
        ),
        findsOneWidget,
      );
      final spinner = tester.getRect(find.byType(CircularProgressIndicator));
      final heading = tester.getRect(
        find.text(
          state.delayed
              ? 'This image is taking longer to straighten'
              : 'Analyzing image…',
        ),
      );
      expect(heading.top - spinner.bottom, 16);
      if (capture != null) {
        await tester.runAsync(() async {
          final image =
              await (boundary.currentContext!.findRenderObject()
                      as RenderRepaintBoundary)
                  .toImage(pixelRatio: 1);
          final bytes = await image.toByteData(format: ui.ImageByteFormat.png);
          await Directory(capture).create(recursive: true);
          await File('$capture/${state.name}.png')
              .writeAsBytes(bytes!.buffer.asUint8List());
          image.dispose();
        });
      }
    }
    await tester.pumpWidget(const SizedBox());
    tester.view.resetPhysicalSize();
    tester.view.resetDevicePixelRatio();
  });
}
