import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:tracker_engine/tracker_engine.dart';

/// Read coordinates without changing their IEEE 754 bits or sample indices.
/// Geometry validation belongs to InteriorMesh for both storage formats.
List<TrackerPoint> readMeshGrid(
  Map<String, dynamic> data,
  int rows,
  int columns,
) {
  if (!data.containsKey('grid_encoding') && !data.containsKey('grid_data')) {
    final grid = data['grid'];
    if (grid is! List) throw const FormatException('Missing dewarping grid');
    return [
      for (final point in grid)
        if (point is List &&
            point.length == 2 &&
            point[0] is num &&
            point[1] is num)
          TrackerPoint(
            (point[0] as num).toDouble(),
            (point[1] as num).toDouble(),
          )
        else
          throw const FormatException('Invalid dewarping grid point'),
    ];
  }
  if (data.containsKey('grid') ||
      data['grid_encoding'] != 'float64le-gzip-base64' ||
      data['grid_data'] is! String) {
    throw const FormatException('Invalid dewarping grid encoding');
  }
  const maxBytes = 512 * 1024 * 1024;
  const maxPoints = maxBytes ~/ 16;
  // Bound each dimension and the product before multiplying native integers.
  if (rows < 2 ||
      columns < 2 ||
      rows > maxPoints ||
      columns > maxPoints ||
      rows > maxPoints ~/ columns) {
    throw const FormatException('Invalid compressed grid dimensions');
  }
  final expected = rows * columns * 16;
  final sink = _GridBytes(expected);
  final decoder = gzip.decoder.startChunkedConversion(sink);
  decoder.add(base64.decode(data['grid_data'] as String));
  decoder.close();
  final bytes = ByteData.sublistView(sink.takeBytes());
  return [
    for (var offset = 0; offset < expected; offset += 16)
      TrackerPoint(
        bytes.getFloat64(offset, Endian.little),
        bytes.getFloat64(offset + 8, Endian.little),
      ),
  ];
}

class _GridBytes implements Sink<List<int>> {
  _GridBytes(this.expected);
  final int expected;
  final _bytes = BytesBuilder(copy: false);
  int _seen = 0;
  bool _closed = false;

  @override
  void add(List<int> bytes) {
    if (_closed || _seen + bytes.length > expected) {
      throw const FormatException('Compressed grid byte length differs');
    }
    _bytes.add(bytes);
    _seen += bytes.length;
  }

  @override
  void close() {
    _closed = true;
    if (_seen != expected) {
      throw const FormatException('Compressed grid byte length differs');
    }
  }

  Uint8List takeBytes() => _bytes.takeBytes();
}
