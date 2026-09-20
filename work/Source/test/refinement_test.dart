import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:tracker_onnx/tracker_onnx.dart';

void main() {
  test('fine localization rejects isolated speckles while retaining a bent edge', () {
    const w=200,h=160;
    final bytes=Uint8List(w*h*3);
    double edge(int y)=>40+(y-80).abs()*.08;
    for(var y=0;y<h;y++){for(var x=0;x<w;x++){
      var color=x>=edge(y)?220:35;
      if((x*73+y*31)%113==0)color=255-color;
      for(var c=0;c<3;c++){bytes[(y*w+x)*3+c]=color;}
    }}
    final points=[for(var y=15;y<=145;y+=5)TrackerPoint((edge(y)+2)/w,y/h)];
    final result=refineFineEdges(points,TrackerRaster(rgbBytes:bytes,width:w,height:h,rowStride:w*3));
    for(var i=2;i<result.length-2;i++){
      expect((result[i].x*w-edge((result[i].y*h).round())).abs(),lessThan(1.6));
    }
  });

  test(
    'pixel evidence rounds a rounded sheet but preserves a square sheet',
    () {
      const w = 500, h = 400;
      const corners = [
        TrackerPoint(.16, .175),
        TrackerPoint(.84, .175),
        TrackerPoint(.84, .825),
        TrackerPoint(.16, .825),
      ];
      for (final radius in [0.0, 24.0]) {
        final bytes = Uint8List(w * h * 3);
        for (var y = 0; y < h; y++) {
          for (var x = 0; x < w; x++) {
            final cx = x.clamp(80 + radius, 420 - radius),
                cy = y.clamp(70 + radius, 330 - radius);
            final inside =
                x >= 80 &&
                x <= 420 &&
                y >= 70 &&
                y <= 330 &&
                ((x - cx) * (x - cx) + (y - cy) * (y - cy) <= radius * radius);
            for (var c = 0; c < 3; c++) {
              bytes[(y * w + x) * 3 + c] = inside ? 235 : 30;
            }
          }
        }
        final raster = TrackerRaster(
          rgbBytes: bytes,
          width: w,
          height: h,
          rowStride: w * 3,
        );
        final segments=<BoundarySegment>[];
      final result = refineDocumentBoundary(raster, corners,segmentOutput:segments);
      if(radius>0){expect(segments.length,8);expect(segments.where((s)=>s.isCurve).length,4);}
        for (final corner in corners) {
          final nearest = result
              .map(
                (p) => math.sqrt(
                  math.pow((p.x - corner.x) * w, 2) +
                      math.pow((p.y - corner.y) * h, 2),
                ),
              )
              .reduce(math.min);
          if (radius == 0) {
            expect(nearest, lessThan(2));
          } else {
            expect(nearest, greaterThan(3));
          }
        }
        expect(
          result.every(
            (p) =>
                p.x.isFinite &&
                p.y.isFinite &&
                p.x >= 0 &&
                p.x <= 1 &&
                p.y >= 0 &&
                p.y <= 1,
          ),
          isTrue,
        );
      }
    },
  );
  test('internal dark heading cannot replace a supported physical edge', () {
    const w = 800, h = 600;
    final bytes = Uint8List(w * h * 3);
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final inside = x >= 80 && x <= 720 && y >= 80 && y <= 520;
        final heading = inside && x > 110 && x < 690 && y >= 100 && y <= 140;
        final color = heading
            ? 25
            : inside
            ? 235
            : 180;
        for (var c = 0; c < 3; c++) {
          bytes[(y * w + x) * 3 + c] = color;
        }
      }
    }
    final result = refineDocumentBoundary(
      TrackerRaster(rgbBytes: bytes, width: w, height: h, rowStride: w * 3),
      const [
        TrackerPoint(.1, 80 / h),
        TrackerPoint(.9, 80 / h),
        TrackerPoint(.9, 520 / h),
        TrackerPoint(.1, 520 / h),
      ],
    );
    final top = result.where((p) => p.x > .2 && p.x < .8 && p.y < .4).toList();
    expect(top, isNotEmpty);
    expect(top.every((p) => (p.y * h - 80).abs() < 3), isTrue);
  });
}
