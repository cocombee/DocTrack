import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
void main(){test('tiny corner lead-ins collapse while a real fold remains',(){
 const p=[TrackerPoint(0,0),TrackerPoint(.98,0),TrackerPoint(1,0),TrackerPoint(1,.02),TrackerPoint(1,1),TrackerPoint(0,1)];
 final edges=[for(var i=0;i<p.length;i++)BoundarySegment(p[i],p[(i+1)%p.length])];
 final result=simplifyCornerAnchors(edges,100,100);
 expect(result.length,4);expect(result[0].end.x,1);expect(result[0].end.y,0);
 const fold=[TrackerPoint(0,0),TrackerPoint(.5,-.1),TrackerPoint(1,0),TrackerPoint(1,1),TrackerPoint(0,1)];
 final bent=[for(var i=0;i<fold.length;i++)BoundarySegment(fold[i],fold[(i+1)%fold.length])];
 expect(simplifyCornerAnchors(bent,100,100).length,5);
});}
