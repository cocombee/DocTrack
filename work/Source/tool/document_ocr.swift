import Foundation
import Vision
import ImageIO

let manifest = URL(fileURLWithPath: CommandLine.arguments[1])
let entries = try JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as! [[String:Any]]
let base = manifest.deletingLastPathComponent()
var output: [String:[String]] = [:]
for entry in entries {
 let url = base.appendingPathComponent(entry["image"] as! String)
 var best:[String] = []
 for orientation in [CGImagePropertyOrientation.up,.right,.down,.left] {
  let request = VNRecognizeTextRequest()
  request.recognitionLevel = .accurate
  request.minimumTextHeight = 0.005
  request.usesLanguageCorrection = false
  if #available(macOS 13.0, *) {request.automaticallyDetectsLanguage = true}
  do {
   try VNImageRequestHandler(url:url,orientation:orientation).perform([request])
   let lines = (request.results ?? []).compactMap {$0.topCandidates(1).first}.filter {$0.confidence > 0.25}.map {$0.string}
   if lines.joined().count > best.joined().count {best=lines}
  } catch {}
 }
 output[entry["name"] as! String]=best
 fputs("OCR \(output.count)/\(entries.count)\n",stderr)
}
try JSONSerialization.data(withJSONObject:output).write(to:URL(fileURLWithPath:CommandLine.arguments[2]))
