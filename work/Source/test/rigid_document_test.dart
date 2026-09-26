import 'package:flutter_test/flutter_test.dart';
import 'package:tracker_engine/tracker_engine.dart';
import 'package:pen_tool/flattening/document_format.dart';
import 'package:pen_tool/flattening/rigid_document.dart';

void main() {
  const corners = [
    TrackerPoint(0, 0),
    TrackerPoint(1, 0),
    TrackerPoint(1, 1),
    TrackerPoint(0, 1),
  ];
  BoundaryResult boundary(List<BoundarySegment> segments) => BoundaryResult(
    width: 600,
    height: 960,
    status: BoundaryStatus.uncertain,
    outerBoundary: [],
    corners: corners,
    boundarySegments: segments,
    confidence: .9,
    modelVersion: 'test',
    preprocessingVersion: 'upright',
  );
  BoundarySegment corner(int index, {bool curved = true}) {
    final right = index % 2 == 1, bottom = index >= 2;
    TrackerPoint p(double x, double y) =>
        TrackerPoint(right ? 1 - x : x, bottom ? 1 - y : y);
    return BoundarySegment(
      p(0, .06),
      p(.06, 0),
      control1: curved ? p(0, .027) : null,
      control2: curved ? p(.027, 0) : null,
    );
  }

  final paper = boundary([
    for (var i = 0; i < 4; i++)
      BoundarySegment(corners[i], corners[(i + 1) % 4]),
  ]);
  final estimated = identifyDocumentFormat(600 / 960, []);

  test('receipt with card-like proportions retains content correction', () {
    expect(estimated.label, startsWith('Card proportions'));
    expect(isRigidDocumentForFlattening(paper, estimated, []), isFalse);
    expect(
      isRigidDocumentForFlattening(paper, estimated, [
        'HÓA ĐƠN BÁN HÀNG',
        '900,000',
        '935,000',
        '35,000',
      ]),
      isFalse,
    );
  });
  test('long bowed paper sides are not rounded corners', () {
    final bowed = boundary([
      BoundarySegment(
        corners[0],
        corners[1],
        control1: const TrackerPoint(.3, .08),
        control2: const TrackerPoint(.7, .08),
      ),
      BoundarySegment(corners[1], corners[2]),
      BoundarySegment(
        corners[2],
        corners[3],
        control1: const TrackerPoint(.7, .94),
        control2: const TrackerPoint(.3, .94),
      ),
      BoundarySegment(corners[3], corners[0]),
    ]);
    expect(isRigidDocumentForFlattening(bowed, estimated, []), isFalse);
  });
  test(
    'two true rounded corners preserve rigid card despite receipt-like OCR',
    () {
      expect(
        isRigidDocumentForFlattening(
          boundary([corner(0), corner(3)]),
          estimated,
          ['invoice total', '900,000', '935,000', '35,000'],
        ),
        isTrue,
      );
    },
  );
  test('two distinct two-point corners preserve rigid card', () {
    expect(
      isRigidDocumentForFlattening(
        boundary([corner(0, curved: false), corner(3, curved: false)]),
        estimated,
        [],
      ),
      isTrue,
    );
    expect(
      isRigidDocumentForFlattening(
        boundary([corner(0, curved: false), corner(0, curved: false)]),
        estimated,
        [],
      ),
      isFalse,
    );
  });
  test('confirmed identity and passport retain rigid behavior', () {
    expect(
      isRigidDocumentForFlattening(
        paper,
        const DocumentFormat('ID / credit-card proportions', 1.585),
        [],
      ),
      isTrue,
    );
    expect(
      isRigidDocumentForFlattening(
        paper,
        const DocumentFormat('Passport', 1.42),
        [],
      ),
      isTrue,
    );
    expect(
      isRigidDocumentForFlattening(
        paper,
        const DocumentFormat('Unidentified', null),
        ['identity nationality surname'],
      ),
      isTrue,
    );
  });
}
