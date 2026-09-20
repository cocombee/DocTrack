import Foundation
import Vision
import ImageIO
let manifest=URL(fileURLWithPath:CommandLine.arguments[1])
let entries=try JSONSerialization.jsonObject(with:Data(contentsOf:manifest)) as! [[String:Any]]
var output:[String:Any]=[:]
for entry in entries {
 let request=VNDetectDocumentSegmentationRequest()
 try VNImageRequestHandler(url:manifest.deletingLastPathComponent().appendingPathComponent(entry["image"] as! String)).perform([request])
 if let q=request.results?.first {
  output[entry["name"] as! String]=[q.topLeft,q.topRight,q.bottomRight,q.bottomLeft].map{["x":$0.x,"y":1-$0.y]}
 }
}
try JSONSerialization.data(withJSONObject:output).write(to:URL(fileURLWithPath:CommandLine.arguments[2]))
