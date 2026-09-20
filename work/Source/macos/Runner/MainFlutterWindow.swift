import Cocoa
import FlutterMacOS
import Vision
import ImageIO

class MainFlutterWindow: NSWindow {
  private let documentTextWorker = DocumentTextWorker()
  override func awakeFromNib() {
    let flutterViewController = FlutterViewController()
    let windowFrame = self.frame
    self.styleMask.insert(.fullSizeContentView)
    self.titleVisibility = .hidden
    self.titlebarAppearsTransparent = true
    // Establish the full-window content area before attaching Flutter. Adding
    // the style afterwards retains the original title-bar-sized view inset.
    self.contentViewController = flutterViewController
    self.setFrame(windowFrame, display: true)
    self.backgroundColor = NSColor(calibratedRed: 0.035, green: 0.035, blue: 0.043, alpha: 1)
    self.isMovableByWindowBackground = false
    RegisterGeneratedPlugins(registry: flutterViewController)
    let channel = FlutterMethodChannel(name: "doctrack/window", binaryMessenger: flutterViewController.engine.binaryMessenger)
    channel.setMethodCallHandler { [weak self] call, result in
      guard let self = self else { result(nil); return }
      switch call.method {
      case "recognizeDocumentText":
        guard let args = call.arguments as? [String: Any], let data = args["data"] as? FlutterStandardTypedData, let id = args["id"] as? Int else { result(FlutterError(code: "arguments", message: "Missing image", details: nil)); return }
        self.documentTextWorker.recognize(data.data, id: id, result: result); return
      case "orientFlattenedDocument":
        guard let args = call.arguments as? [String: Any], let data = args["data"] as? FlutterStandardTypedData, let id = args["id"] as? Int else { result(nil); return }
        self.documentTextWorker.orient(data.data, id: id, result: result); return
      case "cancelDocumentText":
        if let id = call.arguments as? Int { self.documentTextWorker.cancel(id) }; result(nil); return
      case "close": self.performClose(nil)
      case "minimize": self.miniaturize(nil)
      case "maximize": self.zoom(nil)
      default: result(FlutterMethodNotImplemented); return
      }
      result(nil)
    }
    super.awakeFromNib()
    DispatchQueue.main.async { [weak self] in
      for type in [NSWindow.ButtonType.closeButton, .miniaturizeButton, .zoomButton] {
        self?.standardWindowButton(type)?.isHidden = true
      }
    }
  }
}

private final class DocumentTextWorker {
  private let lock = NSLock()
  private var cancelled = Set<Int>()
  private var requests: [Int: VNRequest] = [:]
  func cancel(_ id: Int) {
    lock.lock(); cancelled = cancelled.filter {$0 > id - 128}; cancelled.insert(id); let request = requests[id]; lock.unlock()
    request?.cancel()
  }
  func orient(_ data: Data, id: Int, result: @escaping FlutterResult) {
    DispatchQueue.global(qos: .userInitiated).async {
      let answer = DocumentOrientation.analyze(data, cancelled: {
        self.lock.lock(); defer { self.lock.unlock() }; return self.cancelled.contains(id)
      }, register: { request in
        self.lock.lock(); self.requests[id] = request; self.lock.unlock()
      })
      self.lock.lock(); self.requests.removeValue(forKey: id); self.cancelled.remove(id); self.lock.unlock()
      DispatchQueue.main.async { result(answer) }
    }
  }
  func recognize(_ data: Data, id: Int, result: @escaping FlutterResult) {
    DispatchQueue.global(qos: .userInitiated).async {
      var best: [String] = []
      for orientation in [CGImagePropertyOrientation.up,.right,.down,.left] {
        self.lock.lock(); let stopped = self.cancelled.contains(id); self.lock.unlock()
        if stopped { break }
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.minimumTextHeight = 0.005
        request.usesLanguageCorrection = false
        if #available(macOS 13.0, *) { request.automaticallyDetectsLanguage = true }
        self.lock.lock(); self.requests[id] = request; self.lock.unlock()
        do {
          try VNImageRequestHandler(data: data, orientation: orientation).perform([request])
          let lines = (request.results ?? []).compactMap {$0.topCandidates(1).first}.filter {$0.confidence > 0.25}.map {$0.string}
          if lines.joined().count > best.joined().count { best = lines }
        } catch {}
      }
      self.lock.lock(); self.requests.removeValue(forKey: id); self.cancelled.remove(id); self.lock.unlock()
      DispatchQueue.main.async { result(best) }
    }
  }
}


struct DocumentOrientation {
  static func analyze(_ data: Data, cancelled: () -> Bool = { false }, register: (VNRequest) -> Void = { _ in }) -> [String: Any] {
    let orientations: [CGImagePropertyOrientation] = [.up, .right, .down, .left]
    var scores = [Double](repeating: 0, count: 4)
    var wordCounts = [Int](repeating: 0, count: 4)
    if cancelled() { return ["degrees": 0, "evidence": "cancelled"] }
    let textRequest = VNRecognizeTextRequest()
    textRequest.recognitionLevel = .accurate
    textRequest.minimumTextHeight = 0.005
    textRequest.usesLanguageCorrection = false
    if #available(macOS 13.0, *) { textRequest.automaticallyDetectsLanguage = true }
    register(textRequest)
    try? VNImageRequestHandler(data: data, orientation: .up).perform([textRequest])
    // Vision can recognise text at any rotation. Reading confidence alone
    // therefore cannot establish page orientation; use glyph positions.
    for observation in textRequest.results ?? [] {
      guard let text = observation.topCandidates(1).first, text.confidence >= 0.5 else { continue }
      let letters = text.string.indices.filter { text.string[$0].isLetter }
      guard letters.count >= 3, let a = letters.first, let b = letters.last,
            let first = try? text.boundingBox(for: a..<text.string.index(after: a)),
            let last = try? text.boundingBox(for: b..<text.string.index(after: b)) else { continue }
      let rtl = text.string.unicodeScalars.filter {
        CharacterSet.letters.contains($0) && (0x0590...0x08FF).contains($0.value)
      }.count > letters.count / 2
      let sign = rtl ? -1.0 : 1.0
      let dx = (last.boundingBox.midX - first.boundingBox.midX) * sign
      let dy = (last.boundingBox.midY - first.boundingBox.midY) * sign
      guard hypot(dx, dy) >= 0.02 else { continue }
      let angle = atan2(dy, dx) * 180 / Double.pi
      let quarter = Int((angle / 90).rounded())
      guard abs(angle - Double(quarter) * 90) <= 25 else { continue }
      let i = (quarter + 4) % 4
      wordCounts[i] += 1
      scores[i] += Double(min(letters.count, 40)) * pow(Double(text.confidence), 2)
    }
    let ranked = (0..<4).sorted { scores[$0] > scores[$1] }
    let best = ranked[0]
    let textWinner: Int? = wordCounts[best] >= 2 && scores[best] >= 8 && scores[best] > scores[ranked[1]] * 1.5 ? best : nil
    // Portrait fallback: prefer a detected face with small roll in the
    // candidate orientation. Ambiguous or tilted portraits cause no rotation.
    var faces = [Double](repeating: 0, count: 4)
    for (i, orientation) in orientations.enumerated() {
      if cancelled() { return ["degrees": 0, "evidence": "cancelled"] }
      let request = VNDetectFaceLandmarksRequest(); register(request)
      try? VNImageRequestHandler(data: data, orientation: orientation).perform([request])
      for face in request.results ?? [] {
        guard let roll = face.roll?.doubleValue, abs(roll) < .pi / 6,
              face.landmarks?.leftEye != nil, face.landmarks?.rightEye != nil,
              face.landmarks?.nose != nil else { continue }
        faces[i] = max(faces[i], Double(face.confidence) * (1 - abs(roll) / .pi))
      }
    }
    let order = (0..<4).sorted { faces[$0] > faces[$1] }
    let faceWinner: Int? = faces[order[0]] > 0.7 && faces[order[0]] > faces[order[1]] + 0.15 ? order[0] : nil
    let decision = resolve(text: textWinner, face: faceWinner)
    return ["degrees": decision.degrees, "evidence": decision.evidence, "scores": scores, "faceScores": faces]
  }

  static func resolve(text: Int?, face: Int?) -> (degrees: Int, evidence: String) {
    if let text = text, let face = face, text != face { return (0, "conflicting") }
    if let text = text { return (text * 90, "text") }
    if let face = face { return (face * 90, "face") }
    return (0, "ambiguous")
  }
}
