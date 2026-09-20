import 'package:tracker_engine/tracker_engine.dart';

/// Format evidence is local to flattening and never changes detection.
class DocumentFormat {
  const DocumentFormat(
    this.label,
    this.landscapeRatio, {
    this.widthMm,
    this.heightMm,
  });
  final String label;
  final double? landscapeRatio, widthMm, heightMm;
  double? orientedRatio(double observed) => landscapeRatio == null
      ? null
      : observed >= 1
      ? landscapeRatio
      : 1 / landscapeRatio!;
}

DocumentFormat identifyDocumentFormat(
  double estimatedRatio,
  List<String> lines,
) {
  final ratio = estimatedRatio >= 1 ? estimatedRatio : 1 / estimatedRatio;
  final text = lines.join(' ').toLowerCase();
  bool words(String pattern) =>
      RegExp(pattern, caseSensitive: false).hasMatch(text);
  bool plausible(double target, double tolerance) =>
      (ratio / target - 1).abs() < tolerance;
  // Exact paper sizes need an explicit format clue. Equal aspect ratios do not
  // reveal physical dimensions (A4/A5/B5 all share sqrt(2) proportions).
  if (words(r'\ba4\b') && plausible(297 / 210, .18)) {
    return const DocumentFormat('A4', 297 / 210, widthMm: 297, heightMm: 210);
  }
  if (words(r'\bb5\b') && plausible(250 / 176, .18)) {
    return const DocumentFormat('B5', 250 / 176, widthMm: 250, heightMm: 176);
  }
  if (words(r'\b(letter size|us letter|8[½.]5?\s*[x×]\s*11)\b') &&
      plausible(11 / 8.5, .18)) {
    return const DocumentFormat(
      'Letter',
      11 / 8.5,
      widthMm: 279.4,
      heightMm: 215.9,
    );
  }
  if (words(
        r'\b(passport|passeport|pasaporte|reisepass|pasport|putovnica)\b',
      ) &&
      plausible(125 / 88, .22)) {
    return const DocumentFormat(
      'Passport',
      125 / 88,
      widthMm: 125,
      heightMm: 88,
    );
  }
  if (words(
        r'\b(identity|identification|driver|driving|licence|license|credit|debit|identidad|identita|fuhrerschein|führerschein)\b|주민등록|身份证|身分證',
      ) &&
      plausible(85.6 / 53.98, .22)) {
    return const DocumentFormat(
      'ID / credit-card proportions',
      85.6 / 53.98,
      widthMm: 85.6,
      heightMm: 53.98,
    );
  }
  // A business card is not one worldwide size. Text can identify the family,
  // but should not stretch it into a country's arbitrary standard.
  if (words(r'\b(www\.|https?://)|@') &&
      words(r'\b(tel|mobile|fax|director|manager|consultant)\b') &&
      ratio > 1.35 &&
      ratio < 2.1) {
    return const DocumentFormat('Business card · estimated proportions', null);
  }
  // Ratio-only recognition is deliberately labelled as a family estimate.
  final candidates = <DocumentFormat>[
    const DocumentFormat('A / B paper family · estimated', 1.4142135623730951),
    const DocumentFormat('Letter proportions · estimated', 11 / 8.5),
    const DocumentFormat('Card proportions · estimated', 85.6 / 53.98),
  ];
  for (final format in candidates) {
    if (plausible(format.landscapeRatio!, .025)) return format;
  }
  return const DocumentFormat('Unidentified · estimated proportions', null);
}

/// Physical size is not evidence of rounded corners. Long paper-side curves
/// must not opt a sheet out of rectangular stock repair.
bool preserveDocumentOutline(
  DocumentFormat format,
  DocumentContentEvidence content, {
  required bool hasRoundedCorners,
}) {
  // Format identifies the document family, not its physical corner shape.
  // Preserve rounded stock only when the boundary detector found a coherent
  // rounded-corner family. Passport and ID cards may have hard corners, and a
  // cut/occluded corner must remain eligible for background repair.
  if (!hasRoundedCorners) return false;
  return content.kind == DocumentKind.identityDocument ||
      format.label.startsWith('ID /') ||
      format.label.startsWith('Passport') ||
      format.label.startsWith('Business card') ||
      content.kind != DocumentKind.transactionPaper;
}
