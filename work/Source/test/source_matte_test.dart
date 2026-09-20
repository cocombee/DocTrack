import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:tracker_engine/tracker_engine.dart';
import 'package:pen_tool/flattening/source_matte.dart';
import 'package:pen_tool/flattening/document_format.dart';

void main() {
  List<BoundarySegment> box(double low, double high) {
    final p = [
      TrackerPoint(low, low),
      TrackerPoint(high, low),
      TrackerPoint(high, high),
      TrackerPoint(low, high),
    ];
    return [for (var i = 0; i < 4; i++) BoundarySegment(p[i], p[(i + 1) % 4])];
  }

  img.Image paper() {
    final image = img.Image(width: 200, height: 200, numChannels: 4);
    img.fill(image, color: img.ColorRgba8(25, 30, 35, 255));
    img.fillRect(
      image,
      x1: 20,
      y1: 20,
      x2: 179,
      y2: 179,
      color: img.ColorRgba8(220, 225, 230, 255),
    );
    return image;
  }

  test('continuous outside-coloured fringe is removed before warping', () {
    final image = paper();
    image.setPixelRgba(50, 50, 25, 30, 35, 255);
    final removed = sourceBackgroundFringe(image, box(.09, .91));
    expect(removed, contains(100 * 200 + 19));
    expect(removed, isNot(contains(100 * 200 + 23)));
    expect(removed, isNot(contains(50 * 200 + 50)));
    expect(
      image.getPixel(19, 100).a,
      255,
      reason: 'The diagnostic never edits source pixels',
    );
  });
  test('coherent subpixel edge removes mixed pixels but preserves paper', () {
    final image = paper();
    for (var y = 25; y < 175; y++) {
      image.setPixelRgba(20, y, 150, 155, 160, 255);
    }
    final candidate = sourceBackgroundFringe(
      image,
      box(.10, .90),
      coherentEdges: true,
    );
    expect(candidate, contains(100 * 200 + 20));
    expect(candidate, isNot(contains(100 * 200 + 21)));
  });
  test('textured exterior may surround a continuously smooth paper edge', () {
    final image = paper();
    for (var y = 25; y < 175; y++) {
      for (var x = 10; x < 20; x++) {
        final v = x.isEven ? 30 : 95;
        image.setPixelRgba(x, y, v, v + 5, v + 10, 255);
      }
    }
    final old = sourceBackgroundFringe(image, box(.09, .91));
    final candidate = sourceBackgroundFringe(
      image,
      box(.09, .91),
      texturedExterior: true,
    );
    expect(old, isNot(contains(100 * 200 + 19)));
    expect(candidate, contains(100 * 200 + 19));
    expect(candidate, isNot(contains(100 * 200 + 23)));
  });
  test('short border ink is not a continuous exterior-material strip', () {
    final image = paper();
    img.fillRect(
      image,
      x1: 20,
      y1: 80,
      x2: 22,
      y2: 84,
      color: img.ColorRgba8(25, 30, 35, 255),
    );
    for (final mode in [
      (false, false),
      (true, false),
      (false, true),
      (true, true),
    ]) {
      final removed = sourceBackgroundFringe(
        image,
        box(.10, .90),
        texturedExterior: mode.$1,
        coherentEdges: mode.$2,
      );
      for (var y = 80; y <= 84; y++) {
        for (var x = 20; x <= 22; x++) {
          expect(removed, isNot(contains(y * 200 + x)));
        }
      }
    }
  });
  test(
    'paper followed by an inset printed border is not exterior material',
    () {
      final image = paper();
      img.fillRect(
        image,
        x1: 16,
        y1: 20,
        x2: 19,
        y2: 179,
        color: img.ColorRgba8(220, 225, 230, 255),
      );
      img.drawLine(
        image,
        x1: 20,
        y1: 25,
        x2: 20,
        y2: 174,
        color: img.ColorRgba8(25, 30, 35, 255),
      );
      for (final mode in [
        (false, false),
        (true, false),
        (false, true),
        (true, true),
      ]) {
        final removed = sourceBackgroundFringe(
          image,
          box(.09, .91),
          texturedExterior: mode.$1,
          coherentEdges: mode.$2,
        );
        expect(removed, isNot(contains(100 * 200 + 20)));
      }
    },
  );
  test('image-clipped sides are not trimmed', () {
    final image = paper();
    final removed = sourceBackgroundFringe(image, [
      const BoundarySegment(
        TrackerPoint(.09, .09),
        TrackerPoint(.91, .09),
        isImageClip: true,
      ),
    ]);
    expect(removed, isEmpty);
  });
  test('card-like size alone does not disable paper edge repair', () {
    final format = identifyDocumentFormat(1.58, []);
    expect(
      preserveDocumentOutline(
        format,
        DocumentContentEvidence.fromLines([]),
        hasRoundedCorners: false,
      ),
      isFalse,
    );
    expect(
      preserveDocumentOutline(
        format,
        DocumentContentEvidence.fromLines([]),
        hasRoundedCorners: true,
      ),
      isTrue,
    );
    expect(
      preserveDocumentOutline(
        format,
        DocumentContentEvidence.fromLines(['identity', 'nationality']),
        hasRoundedCorners: false,
      ),
      isTrue,
    );
  });
}
