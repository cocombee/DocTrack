// Independent word-envelope evidence from the same local Vision OCR already
// used by DocTrack. This research tool neither uploads images nor adds models.
import Foundation
import Vision
import ImageIO

guard CommandLine.arguments.count == 3 else {
    fputs("Usage: document_word_ocr image.png words.json\n", stderr)
    exit(2)
}
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.minimumTextHeight = 0.005
request.usesLanguageCorrection = false
if #available(macOS 13.0, *) { request.automaticallyDetectsLanguage = true }
try VNImageRequestHandler(url: URL(fileURLWithPath: CommandLine.arguments[1]),
                         orientation: .up).perform([request])
let tokens = try NSRegularExpression(pattern: "\\S+")
var output: [[String: Any]] = []
for observation in request.results ?? [] {
    guard let recognized = observation.topCandidates(1).first,
          recognized.confidence > 0.25 else { continue }
    let text = recognized.string
    for token in tokens.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
        guard let range = Range(token.range, in: text),
              let quad = try recognized.boundingBox(for: range) else { continue }
        let points = [quad.topLeft, quad.topRight, quad.bottomRight, quad.bottomLeft]
        output.append([
            "text": String(text[range]),
            "confidence": recognized.confidence,
            "points": points.map { [Double($0.x), Double(1 - $0.y)] }
        ])
    }
}
try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys])
    .write(to: URL(fileURLWithPath: CommandLine.arguments[2]), options: .atomic)
