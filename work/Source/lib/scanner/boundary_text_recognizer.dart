import 'package:flutter/services.dart';
import 'package:tracker_engine/tracker_engine.dart';

import '../flattening/document_words.dart';

/// Reads photo text once for boundary classification.
///
/// The upright path is opt-in while it is being qualified. It may decide a
/// transaction only from its own clear content evidence; all other pages use
/// the established full-orientation Vision path.
Future<List<String>?> recognizeBoundaryText(
  Uint8List encoded, {
  required int id,
  MethodChannel channel = const MethodChannel('doctrack/window'),
  bool fastTransaction = const bool.fromEnvironment(
    'DOCTRACK_FAST_TRANSACTION_OCR',
  ),
  void Function(List<DocumentWordGeometry>)? onWords,
  void Function()? checkCancelled,
}) async {
  final arguments = {'data': encoded, 'id': id};
  if (fastTransaction) {
    try {
      List<String>? upright;
      List<DocumentWordGeometry>? words;
      if (onWords == null) {
        upright = await channel.invokeListMethod<String>(
          'recognizeDocumentTextUpright',
          arguments,
        );
      } else {
        final evidence = await channel.invokeMapMethod<String, Object?>(
          'recognizeDocumentTextUprightWithWords',
          arguments,
        );
        final rawLines = evidence?['lines'], rawWords = evidence?['words'];
        if (rawLines is List &&
            rawLines.every((line) => line is String) &&
            rawWords is List) {
          upright = rawLines.cast<String>();
          words = rawWords
              .map(DocumentWordGeometry.fromNative)
              .whereType<DocumentWordGeometry>()
              .toList();
        }
      }
      if (DocumentContentEvidence.fromLines(upright ?? const []).kind ==
          DocumentKind.transactionPaper) {
        checkCancelled?.call();
        if (words != null) onWords?.call(List.unmodifiable(words));
        return upright;
      }
    } on MissingPluginException {
      /* An older native host still supplies the established OCR path. */
    } on PlatformException {
      /* Inconclusive quick OCR falls back to the established path. */
    }
  }
  checkCancelled?.call();
  return channel.invokeListMethod<String>('recognizeDocumentText', arguments);
}
