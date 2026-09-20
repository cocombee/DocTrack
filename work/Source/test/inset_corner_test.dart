import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
void main(){
 test('inset collapses a consumed corner without crossing or changing large arcs',(){
  for(final radius in [2.0,8.0]){
   const corners=[TrackerPoint(.1,.1),TrackerPoint(.9,.1),TrackerPoint(.9,.9),TrackerPoint(.1,.9)];
   final arcs=<BoundarySegment>[];
   for(var i=0;i<4;i++){
    final p=corners[i],prev=corners[(i+3)%4],next=corners[(i+1)%4];
    TrackerPoint toward(TrackerPoint q)=>TrackerPoint(p.x+(q.x-p.x)*radius/80,p.y+(q.y-p.y)*radius/80);
    final a=toward(prev),b=toward(next);
    arcs.add(BoundarySegment(a,b,control1:TrackerPoint(a.x+(p.x-a.x)*2/3,a.y+(p.y-a.y)*2/3),control2:TrackerPoint(b.x+(p.x-b.x)*2/3,b.y+(p.y-b.y)*2/3)));
   }
   final path=[for(var i=0;i<4;i++)...[arcs[i],BoundarySegment(arcs[i].end,arcs[(i+1)%4].start)]];
   final result=offsetBoundarySegments(path,100,100,4);
   expect(result.length,radius<4?4:8);
   if(radius<4){expect(result.every((s)=>!s.isCurve),isTrue);for(final s in result){expect(s.start.x>=.14-1e-9&&s.start.x<=.86+1e-9,isTrue);expect(s.start.y>=.14-1e-9&&s.start.y<=.86+1e-9,isTrue);}}
   for(var i=0;i<result.length;i++){final a=result[i].end,b=result[(i+1)%result.length].start;expect(a.x,closeTo(b.x,1e-12));expect(a.y,closeTo(b.y,1e-12));}
   expect(path.length,8);
  }
 });
}
