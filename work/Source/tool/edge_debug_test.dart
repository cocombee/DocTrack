import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:pen_tool/flattening/background_surface.dart';
void main() { test('inspect local edge evidence', () {
 final im=img.decodePng(File('../Scanner Integration/Flattening Review 46/Image 2 Raw.png').readAsBytesSync())!;
 final surface=BackgroundSurface.estimate(im);
 for(final x in [50,100,150,200,250,300,350,380]) {
   stdout.writeln('$x: ${surface.color(x/im.width,0)}');
 }
 for(final y in [630,632,634,636,638,639]) { stdout.writeln('bottom $y: ${surface.color(208/im.width,y/im.height)} pixel ${im.getPixel(208,y)}'); }
 final cut=missingEdgeFringe(im,surface);
 stdout.writeln('Repair pixels: ${cut.length}, columns ${cut.map((i)=>i%im.width).toSet()}');
}); }
