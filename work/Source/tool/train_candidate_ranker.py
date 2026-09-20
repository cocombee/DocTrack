"""Offline training only. Runtime receives numeric feature weights, never labels or image IDs."""
from pathlib import Path
import json
import numpy as np
root=Path(__file__).resolve().parents[2]
rows=json.loads((root/'Scanner Integration/Candidate Scores V7.json').read_text())
rows=[r for r in rows if r['split']=='train']
x=np.array([r['features'] for r in rows]); y=np.array([r['iou'] for r in rows])
z=np.column_stack([np.ones(len(x)),x]+[x[:,i]*x[:,j] for i in range(11) for j in range(i,11)])
weights=np.linalg.solve(z.T@z+np.eye(z.shape[1]),z.T@y)
path=root/'Tracker Engine/lib/src/ranker.dart'
path.write_text('// Learned only from the frozen training split. No per-image lookup.\nconst candidateRankerWeights = <double>['+','.join(format(float(v),'.12g') for v in weights)+'];\ndouble rankCandidate(List<double> x) {\n  var score=candidateRankerWeights[0]; var k=1;\n  for(final v in x) { score+=v*candidateRankerWeights[k++]; }\n  for(var i=0;i<x.length;i++) { for(var j=i;j<x.length;j++) { score+=x[i]*x[j]*candidateRankerWeights[k++]; } }\n  return score;\n}\n')
(root/'Scanner Integration/Ranker Training.json').write_text(json.dumps({'training_images':sorted(set(r['index'] for r in rows)),'candidate_count':len(rows),'features':11,'ridge':1,'source':'Candidate Scores V7','weights':weights.tolist()},indent=2))
print('Exported training-only candidate ranker')
