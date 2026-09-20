import 'package:flutter_test/flutter_test.dart';
import 'run_onnx_tracker.dart' as runner;
void main() {
 test('evaluate native ONNX adapter against all image pixels',() async {
  await runner.main(['../Scanner Integration/Corner Focus Inputs','../Scanner Integration/DocAligner Evaluation/lcnet100.onnx','../Scanner Integration/DocAligner Evaluation/lcnet050.onnx','../Scanner Integration/Predictions Corner V15']);
 },timeout:const Timeout(Duration(minutes:3)));
}
