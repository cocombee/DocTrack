import 'package:flutter_test/flutter_test.dart';
import 'run_onnx_tracker.dart' as runner;
void main() {
 test('evaluate native ONNX adapter against all image pixels',() async {
  await runner.main(['../Data/originals','../Scanner Integration/DocAligner Evaluation/lcnet100.onnx','../Scanner Integration/DocAligner Evaluation/lcnet050.onnx','../Scanner Integration/Predictions Simple V17']);
 },timeout:const Timeout(Duration(minutes:3)));
}
