import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';

void main() {
  test('punctuated numbers and dates do not identify a receipt', () {
    expect(
      DocumentContentEvidence.fromLines([
        '12.03.1990',
        '12/03/2020',
        '1,234.00',
        '25.00',
        '99.50',
      ]).kind,
      DocumentKind.unknown,
    );
  });
  test(
    'transaction label with repeated amount rows identifies transaction paper',
    () {
      expect(
        DocumentContentEvidence.fromLines([
          'TOTAL',
          '1,234.00',
          '25.00',
          '99.50',
        ]).kind,
        DocumentKind.transactionPaper,
      );
    },
  );
  test('ID labels outweigh incidental punctuated numbers', () {
    expect(
      DocumentContentEvidence.fromLines([
        'IDENTITY CARD',
        'NATIONALITY',
        'DATE OF BIRTH',
        '1,234.00',
        '25.00',
        '99.50',
      ]).kind,
      DocumentKind.identityDocument,
    );
  });
  test('ordinary article remains unknown', () {
    expect(
      DocumentContentEvidence.fromLines([
        'Active in Cape Town',
        'Date of birth of Cape Town',
        '1652',
        'Population',
        '3,400,000',
      ]).kind,
      DocumentKind.unknown,
    );
  });
  test('Vietnamese card back needs multiple identity fields', () {
    expect(
      DocumentContentEvidence.fromLines(['Đặc điểm nhận dạng', 'CỤC CẢNH SÁT'])
          .kind,
      DocumentKind.identityDocument,
    );
    expect(
      DocumentContentEvidence.fromLines(['Bản tin cảnh sát']).kind,
      DocumentKind.unknown,
    );
  });
}
