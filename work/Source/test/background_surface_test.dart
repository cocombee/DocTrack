import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:pen_tool/flattening/background_surface.dart';

void main() {
  test('missing edge fringe is repaired without erasing isolated ink', () {
    final source = img.Image(width: 400, height: 400, numChannels: 4);
    img.fill(source, color: img.ColorRgba8(190, 200, 210, 255));
    final surface = BackgroundSurface.estimate(source);
    for (var x = 170; x < 195; x++) {
      source.setPixelRgba(x, 399, 0, 0, 0, 0);
    }
    img.fillRect(
      source,
      x1: 195,
      y1: 394,
      x2: 199,
      y2: 399,
      color: img.ColorRgba8(30, 35, 40, 255),
    );
    img.fillRect(
      source,
      x1: 240,
      y1: 394,
      x2: 244,
      y2: 399,
      color: img.ColorRgba8(0, 0, 0, 255),
    );
    img.fillRect(
      source,
      x1: 179,
      y1: 370,
      x2: 182,
      y2: 382,
      color: img.ColorRgba8(0, 0, 0, 255),
    );
    final fringe = missingEdgeFringe(source, surface);
    expect(fringe, contains(397 * 400 + 197));
    expect(fringe, isNot(contains(397 * 400 + 242)));
    expect(fringe, isNot(contains(378 * 400 + 180)));
  });
  test('wider fringe keeps strokes continuing into document content', () {
    final source = img.Image(width: 400, height: 400, numChannels: 4);
    img.fill(source, color: img.ColorRgba8(220, 210, 205, 255));
    final surface = BackgroundSurface.estimate(source);
    source.setPixelRgba(100, 399, 0, 0, 0, 0);
    img.fillRect(
      source,
      x1: 101,
      y1: 350,
      x2: 104,
      y2: 399,
      color: img.ColorRgba8(0, 0, 0, 255),
    );
    final fringe = missingEdgeFringe(source, surface, depthFraction: .06);
    expect(fringe, isNot(contains(398 * 400 + 101)));
    expect(fringe, isNot(contains(379 * 400 + 102)));
  });
  test('extended dark fringe cleanup preserves bright fold highlights', () {
    final source = img.Image(width: 400, height: 400, numChannels: 4);
    img.fill(source, color: img.ColorRgba8(190, 190, 190, 255));
    final surface = BackgroundSurface.estimate(source);
    source.setPixelRgba(100, 399, 0, 0, 0, 0);
    img.fillRect(
      source,
      x1: 101,
      y1: 380,
      x2: 104,
      y2: 399,
      color: img.ColorRgba8(245, 245, 245, 255),
    );
    final fringe = missingEdgeFringe(
      source,
      surface,
      depthFraction: .06,
      darkOnly: true,
    );
    expect(fringe, isEmpty);
  });
  test(
    'edge repair follows local stock colour instead of page-wide colour',
    () {
      final source = img.Image(width: 200, height: 200, numChannels: 4);
      img.fill(source, color: img.ColorRgba8(240, 240, 240, 255));
      img.fillRect(
        source,
        x1: 0,
        y1: 0,
        x2: 80,
        y2: 199,
        color: img.ColorRgba8(120, 170, 210, 255),
      );
      final surface = BackgroundSurface.estimate(source);
      expect(surface.color(0, .5), [120, 170, 210]);
      expect(surface.color(1, .5), [240, 240, 240]);
    },
  );
  test(
    'coherent exterior strip is repairable but isolated ink is retained',
    () {
      final source = img.Image(width: 200, height: 200, numChannels: 4);
      img.fill(source, color: img.ColorRgba8(230, 230, 230, 255));
      final surface = BackgroundSurface.estimate(source);
      img.fillRect(
        source,
        x1: 30,
        y1: 0,
        x2: 170,
        y2: 3,
        color: img.ColorRgba8(60, 40, 20, 255),
      );
      img.fillRect(
        source,
        x1: 0,
        y1: 80,
        x2: 3,
        y2: 87,
        color: img.ColorRgba8(0, 0, 0, 255),
      );
      final rejected = exteriorStrips(source, surface);
      expect(rejected, contains(100));
      expect(rejected, isNot(contains(80 * 200)));
    },
  );

  test('colour shading is reconstructed without copying printed strokes', () {
    final source = img.Image(width: 200, height: 120, numChannels: 4);
    for (var y = 0; y < 120; y++) {
      for (var x = 0; x < 200; x++) {
        source.setPixelRgba(
          x,
          y,
          160 + x ~/ 10,
          180 + x ~/ 10,
          200 + x ~/ 10,
          255,
        );
      }
    }
    for (var x = 20; x < 180; x += 12) {
      img.drawRect(
        source,
        x1: x,
        y1: 40,
        x2: x + 5,
        y2: 70,
        color: img.ColorRgb8(0, 0, 0),
        thickness: 2,
      );
    }
    final surface = BackgroundSurface.estimate(source);
    for (var x = 0; x < 200; x++) {
      final color = surface.color(x / 200, .5);
      expect(color[0], closeTo(160 + x / 10, 2));
      expect(color[1], closeTo(180 + x / 10, 2));
    }
  });
  test(
    'sparse cut corner uses nearby shaded stock rather than page average',
    () {
      final source = img.Image(width: 400, height: 600, numChannels: 4);
      img.fill(source, color: img.ColorRgba8(212, 170, 184, 255));
      img.fillRect(
        source,
        x1: 0,
        y1: 0,
        x2: 399,
        y2: 100,
        color: img.ColorRgba8(183, 140, 156, 255),
      );
      img.fillRect(
        source,
        x1: 0,
        y1: 0,
        x2: 65,
        y2: 65,
        color: img.ColorRgba8(0, 0, 0, 0),
      );
      final original = source.getBytes().toList();
      final surface = BackgroundSurface.estimate(source, localFallback: true);
      expect(surface.color(0, 0), [183, 140, 156]);
      expect(
        source.getBytes(),
        original,
        reason: 'Estimating stock must not change pixels or text',
      );
    },
  );
  test('dark stock is not replaced with white', () {
    final source = img.Image(width: 100, height: 100, numChannels: 4);
    img.fill(source, color: img.ColorRgba8(30, 55, 75, 255));
    expect(BackgroundSurface.estimate(source).color(.5, .5), [30, 55, 75]);
  });
  test(
    'feather softens background seam and protects text and transparency',
    () {
      final source = img.Image(width: 400, height: 400, numChannels: 4);
      img.fill(source, color: img.ColorRgba8(210, 220, 230, 255));
      final surface = BackgroundSurface.estimate(source);
      // A small local stock-colour variation at the repaired edge.
      img.fillRect(
        source,
        x1: 0,
        y1: 0,
        x2: 30,
        y2: 399,
        color: img.ColorRgba8(216, 226, 236, 255),
      );
      final repaired = <int>[];
      for (var y = 0; y < 400; y++) {
        for (var x = 0; x < 10; x++) {
          source.setPixelRgba(x, y, 210, 220, 230, 255);
          repaired.add(y * 400 + x);
        }
      }
      img.drawLine(
        source,
        x1: 10,
        y1: 100,
        x2: 20,
        y2: 100,
        color: img.ColorRgb8(10, 10, 10),
      );
      source.setPixelRgba(399, 399, 0, 0, 0, 0);
      featherBackground(source, repaired, surface);
      expect(source.getPixel(10, 200).r, lessThan(216));
      expect(
        source.getPixel(12, 200).r,
        greaterThan(source.getPixel(10, 200).r),
      );
      expect(source.getPixel(20, 200).r, 216);
      expect(source.getPixel(10, 100).r, 10);
      expect(source.getPixel(399, 399).a, 0);
    },
  );
  test('adjacent stock extends local shading without copying ink or changing source', () {
    final source = img.Image(width: 120, height: 120, numChannels: 4);
    img.fill(source, color: img.ColorRgba8(220, 225, 230, 255));
    final surface = BackgroundSurface.estimate(source);
    for (var y = 0; y < 120; y++) {
      for (var x = 10; x < 30; x++) {
        source.setPixelRgba(x, y, 200, 205, 210, 255);
      }
    }
    img.drawLine(
      source,
      x1: 10,
      y1: 60,
      x2: 22,
      y2: 60,
      color: img.ColorRgb8(10, 10, 10),
    );
    final original = img.Image.from(source);
    final repaired = [
      for (var y = 0; y < 120; y++)
        for (var x = 0; x < 10; x++) y * 120 + x,
    ];
    extendLocalStock(source, repaired, surface);
    expect(source.getPixel(9, 30).r, 200);
    expect(source.getPixel(0, 30).r, 200);
    expect(source.getPixel(9, 60).r, 200);
    for (var y = 0; y < 120; y++) {
      for (var x = 10; x < 120; x++) {
        expect(
          source.getPixel(x, y).toList(),
          original.getPixel(x, y).toList(),
        );
      }
    }
  });
}
