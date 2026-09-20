import 'dart:convert';
import 'dart:ui' as ui;
import 'dart:ui' show PointerDeviceKind;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/main.dart';

void main() {
  testWidgets('flat mode persists across selection and cancels pending preparation', (tester) async {
    await tester.runAsync(() async {
      await tester.pumpWidget(const PenToolApp());
      await Future<void>.delayed(const Duration(milliseconds: 250));
    });
    await tester.pumpAndSettle();
    final overlay = find.byWidgetPredicate((w) => w is CustomPaint && w.painter.runtimeType.toString() == '_OverlayPainter');
    final dynamic painter = tester.widget<CustomPaint>(overlay).painter;
    final store = painter.store as Store;
    store.detecting = true;
    await tester.tap(find.byTooltip('Flatten document'));
    await tester.pump();
    expect(find.byTooltip('Save flattened PNG'), findsOneWidget);
    store.index = 1;
    store.refresh();
    await tester.pump();
    expect(find.byTooltip('Save flattened PNG'), findsOneWidget);
    expect(overlay, findsNothing);
    await tester.tap(find.byTooltip('Flatten document'));
    await tester.pump();
    expect(find.byTooltip('Save flattened PNG'), findsNothing);
    expect(overlay, findsOneWidget);
    store.detecting = false;
    expect(tester.takeException(), isNull);
  });
  testWidgets('engine scan keeps the canvas free of status bars', (
    tester,
  ) async {
    await tester.runAsync(() async {
      await tester.pumpWidget(const PenToolApp());
      await Future<void>.delayed(const Duration(milliseconds: 250));
    });
    await tester.pumpAndSettle();
    final overlay = find.byWidgetPredicate(
      (w) =>
          w is CustomPaint &&
          w.painter.runtimeType.toString() == '_OverlayPainter',
    );
    final dynamic painter = tester.widget<CustomPaint>(overlay).painter;
    final Store store = painter.store as Store;
    store.detecting = true;
    store.refresh();
    await tester.pump();
    final footer = find.byKey(const ValueKey('engine-status-footer'));
    expect(footer, findsNothing);
    expect(tester.takeException(), isNull);
    store.detecting = false;
  });
  testWidgets('arrows navigate previous next first and last without wrapping', (
    tester,
  ) async {
    await tester.runAsync(() async {
      await tester.pumpWidget(const PenToolApp());
      await Future<void>.delayed(const Duration(milliseconds: 250));
    });
    await tester.pumpAndSettle();
    final overlay = find.byWidgetPredicate(
      (w) =>
          w is CustomPaint &&
          w.painter.runtimeType.toString() == '_OverlayPainter',
    );
    final dynamic painter = tester.widget<CustomPaint>(overlay).painter;
    final total = (painter.store as Store).samples.length;
    expect(total, greaterThan(1));
    Future<void> key(LogicalKeyboardKey key, int index) async {
      await tester.runAsync(() async {
        await tester.sendKeyEvent(key);
        await Future<void>.delayed(const Duration(milliseconds: 100));
      });
      await tester.pumpAndSettle();
      expect(
        find.byWidgetPredicate(
          (w) => w is Text && (w.data ?? '').startsWith('$index/$total —'),
        ),
        findsOneWidget,
      );
    }

    await key(LogicalKeyboardKey.arrowRight, 2);
    await key(LogicalKeyboardKey.arrowLeft, 1);
    await key(LogicalKeyboardKey.arrowLeft, 1);
    await key(LogicalKeyboardKey.arrowDown, total);
    await key(LogicalKeyboardKey.arrowRight, total);
    await key(LogicalKeyboardKey.arrowUp, 1);
  });
  testWidgets('drag selects anchors and toolbar reveals every selected curve', (
    tester,
  ) async {
    tester.view.physicalSize = const Size(800, 632);
    tester.view.devicePixelRatio = 1;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);
    await tester.runAsync(() async {
      await tester.pumpWidget(const PenToolApp());
      await Future<void>.delayed(const Duration(milliseconds: 250));
    });
    await tester.pumpAndSettle();
    final overlay = find.byWidgetPredicate(
      (w) =>
          w is CustomPaint &&
          w.painter.runtimeType.toString() == '_OverlayPainter',
    );
    expect(overlay, findsOneWidget);
    final dynamic painter = tester.widget<CustomPaint>(overlay).painter;
    final Store store = painter.store as Store;
    final original = jsonEncode(store.points.map((p) => p.toJson()).toList());
    expect(store.points.length, greaterThanOrEqualTo(4));
    final rect = tester.getRect(overlay);
    final start = rect.topLeft + Offset(rect.width * .25, rect.height * .12);
    final end = rect.topLeft + Offset(rect.width * .85, rect.height * .94);
    await tester.dragFrom(start, end - start, kind: PointerDeviceKind.mouse);
    await tester.pumpAndSettle();
    expect(store.multiSelect.length, store.points.length);
    expect(store.dirty, false);
    await tester.tap(find.byTooltip('Reveal reference and its curve handles'));
    await tester.pumpAndSettle();
    expect(store.revealed, store.multiSelect);
    expect(jsonEncode(store.points.map((p) => p.toJson()).toList()), original);
    expect(store.dirty, false);
    expect(tester.takeException(), isNull);
  });
  testWidgets(
    'zoom keeps painted markers and strokes at constant screen size',
    (tester) async {
      await tester.runAsync(() async {
        await tester.pumpWidget(const PenToolApp());
        await Future<void>.delayed(const Duration(milliseconds: 250));
      });
      await tester.pumpAndSettle();
      final overlay = find.byWidgetPredicate(
        (w) =>
            w is CustomPaint &&
            w.painter.runtimeType.toString() == '_OverlayPainter',
      );
      final dynamic painter = tester.widget<CustomPaint>(overlay).painter;
      final Store store = painter.store as Store;
      final Size fit = painter.fit as Size;
      store.points = [
        PenPoint(x: .2, y: .2),
        PenPoint(x: .8, y: .2),
        PenPoint(x: .8, y: .8),
        PenPoint(x: .2, y: .8),
      ];
      for (final smooth in [false, true]) {
        for (final p in store.points) {
          p.smooth = smooth;
          p.hIn = [p.x, p.y];
          p.hOut = [p.x, p.y];
        }
        List<int>? baseline;
        for (final zoom in [1.0, 2.0, 6.0]) {
          store.zoom = zoom;
          final recorder = ui.PictureRecorder();
          final canvas = Canvas(recorder);
          canvas.translate(
            40 - .2 * fit.width * zoom,
            40 - .2 * fit.height * zoom,
          );
          canvas.scale(zoom);
          painter.paint(canvas, fit);
          final picture = recorder.endRecording();
          final bytes = await tester.runAsync(() async {
            final image = await picture.toImage(100, 100);
            final data = await image.toByteData(
              format: ui.ImageByteFormat.rawRgba,
            );
            image.dispose();
            return data!;
          });
          final pixels = bytes!.buffer.asUint8List();
          var white = 0, green = 0;
          for (var x = 20; x < 60; x++) {
            final k = (40 * 100 + x) * 4;
            if (pixels[k] > 230 && pixels[k + 1] > 230 && pixels[k + 2] > 230) {
              white++;
            }
          }
          for (var y = 20; y < 60; y++) {
            final k = (y * 100 + 80) * 4;
            if (pixels[k + 3] > 200 && pixels[k + 1] > pixels[k] * 2) green++;
          }
          final counts = [white, green];
          baseline ??= counts;
          expect(
            counts,
            baseline,
            reason: 'marker/stroke size at $zoom x, smooth=$smooth',
          );
          expect(white, inInclusiveRange(8, 10));
          expect(green, 2);
          picture.dispose();
        }
      }
      final curve = tester.getSize(
        find.byTooltip('Reveal reference and its curve handles'),
      );
      final reload = tester.getSize(find.byTooltip('Reload label from disk'));
      expect(curve, const Size(36, 36));
      expect(reload, curve);
    },
  );

  testWidgets(
    'visible window controls invoke their corresponding macOS actions',
    (tester) async {
      final calls = <String>[];
      tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
        const MethodChannel('doctrack/window'),
        (call) async {
          calls.add(call.method);
          return null;
        },
      );
      addTearDown(
        () => tester.binding.defaultBinaryMessenger.setMockMethodCallHandler(
          const MethodChannel('doctrack/window'),
          null,
        ),
      );
      await tester.runAsync(() async {
        await tester.pumpWidget(const PenToolApp());
        await Future<void>.delayed(const Duration(milliseconds: 250));
      });
      await tester.pumpAndSettle();
      for (final label in [
        'Close window',
        'Minimize window',
        'Maximize or restore window',
      ]) {
        final control = find.byTooltip(label);
        expect(control, findsOneWidget);
        expect(tester.getRect(control).top, inInclusiveRange(6, 12));
        await tester.tap(control);
        await tester.pump();
      }
      expect(calls, ['close', 'minimize', 'maximize']);
    },
  );
}
