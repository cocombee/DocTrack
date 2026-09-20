import 'package:flutter_test/flutter_test.dart';
import 'run_onnx_tracker.dart' as runner;
void main() {
  test('evaluate receipt corner candidate with OCR evidence', () async {
    const selection = String.fromEnvironment('SELECTION',
      defaultValue: '../Data/manifest.json');
    await runner.main([
      '../Data/originals',
      '../Scanner Integration/DocAligner Evaluation/lcnet100.onnx',
      '../Scanner Integration/DocAligner Evaluation/lcnet050.onnx',
      '../Scanner Integration/Predictions Corner Refinement 45',
      '/tmp/doctrack-ocr-lines.json', selection,
    ]);
  }, timeout: const Timeout(Duration(minutes: 5)));
}
