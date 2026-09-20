import 'dart:math' as math;

import 'package:image/image.dart' as img;

/// A field of smooth local colour planes, never generated or copied detail.
/// Only low-frequency surface statistics are extended into missing pixels.
/// Original pixels never enter the repair output except as colour statistics.
class BackgroundSurface {
  BackgroundSurface._(this.channels, [this.local]);
  final List<List<double>>? local;
  static const gridSize = 17;
  final List<List<double>> channels;

  factory BackgroundSurface.estimate(
    img.Image image, {
    bool localFallback = false,
  }) {
    final samples = <List<double>>[];
    final step = math.max(4, math.min(image.width, image.height) ~/ 40);
    final radius = math.max(1, step ~/ 3);
    for (var y = radius; y < image.height - radius; y += step) {
      for (var x = radius; x < image.width - radius; x += step) {
        final colors = <List<double>>[];
        for (var dy = -radius; dy <= radius; dy += radius) {
          for (var dx = -radius; dx <= radius; dx += radius) {
            final p = image.getPixel(x + dx, y + dy);
            if (p.a == 255) {
              colors.add([p.r.toDouble(), p.g.toDouble(), p.b.toDouble()]);
            }
          }
        }
        if (colors.length != 9) continue;
        final mean = List.generate(
          3,
          (c) => colors.fold<double>(0, (s, p) => s + p[c]) / 9,
        );
        final variance =
            colors.fold<double>(
              0,
              (s, p) =>
                  s +
                  List.generate(
                    3,
                    (c) => math.pow(p[c] - mean[c], 2),
                  ).reduce((a, b) => a + b),
            ) /
            27;
        // Exclude strokes, photographs and detailed pattern patches.
        if (variance > 20) continue;
        samples.add([x / image.width, y / image.height, ...mean]);
      }
    }
    if (samples.isEmpty) {
      // Ambiguous detail provides no evidence for a background reconstruction.
      return BackgroundSurface._(List.generate(3, (_) => [255, 0, 0]));
    }
    // Prefer the dominant smooth surface, including dark/coloured documents.
    // No brightness ranking that would turn coloured stock into white paper.
    final bins = <int, List<List<double>>>{};
    for (final s in samples) {
      final key = (s[2] ~/ 32) * 64 + (s[3] ~/ 32) * 8 + s[4] ~/ 32;
      (bins[key] ??= []).add(s);
    }
    final mode = bins.values.reduce((a, b) => a.length >= b.length ? a : b);
    final center = List.generate(
      3,
      (c) => mode.fold<double>(0, (v, s) => v + s[c + 2]) / mode.length,
    );
    var selected = samples
        .where(
          (s) =>
              List.generate(
                3,
                (c) => (s[c + 2] - center[c]).abs(),
              ).reduce(math.max) <=
              48,
        )
        .toList();
    var fit = List.generate(3, (c) => [center[c], 0.0, 0.0]);
    for (var iteration = 0; iteration < 3; iteration++) {
      if (selected.length < 12) break;
      final next = List.generate(3, (c) => _fit(selected, c));
      if (next.any((v) => v == null)) break;
      fit = next.cast<List<double>>();
      final retained = selected
          .where(
            (s) =>
                List.generate(
                  3,
                  (c) =>
                      (s[c + 2] -
                              (fit[c][0] + fit[c][1] * s[0] + fit[c][2] * s[1]))
                          .abs(),
                ).reduce(math.max) <
                12,
          )
          .toList();
      if (retained.length == selected.length) break;
      selected = retained;
    }
    // Limit extrapolation so a small surviving fragment cannot drive extremes.
    for (var c = 0; c < 3; c++) {
      final slope = fit[c][1].abs() + fit[c][2].abs();
      if (slope > 64) fit[c] = [center[c], 0, 0];
    }
    final field = <List<double>>[];
    final short = math.min(image.width, image.height);
    for (var gy = 0; gy < gridSize; gy++) {
      for (var gx = 0; gx < gridSize; gx++) {
        final u = gx / (gridSize - 1), v = gy / (gridSize - 1);
        var nearby = samples
            .where(
              (p) =>
                  math.pow((p[0] - u) * image.width / short, 2) +
                      math.pow((p[1] - v) * image.height / short, 2) <
                  .08 * .08,
            )
            .toList();
        if (localFallback && nearby.length < 3) {
          // A cut corner may contain too few smooth samples. Continue from
          // the closest observed stock, not the page-wide colour plane.
          double distance(List<double> p) =>
              math.pow((p[0] - u) * image.width, 2).toDouble() +
              math.pow((p[1] - v) * image.height, 2).toDouble();
          final nearest = List<List<double>>.from(samples)
            ..sort((a, b) => distance(a).compareTo(distance(b)));
          nearby = nearest.take(12).toList();
        }
        if (nearby.length < (localFallback ? 3 : 6)) {
          field.add([for (final c in fit) c[0] + c[1] * u + c[2] * v]);
          continue;
        }
        final buckets = <int, List<List<double>>>{};
        for (final p in nearby) {
          final key = (p[2] ~/ 32) * 64 + (p[3] ~/ 32) * 8 + p[4] ~/ 32;
          (buckets[key] ??= []).add(p);
        }
        final dominant = buckets.values.reduce(
          (a, b) => a.length >= b.length ? a : b,
        );
        final mean = List.generate(
          3,
          (c) =>
              dominant.fold<double>(0, (a, p) => a + p[c + 2]) /
              dominant.length,
        );
        final clean = nearby
            .where(
              (p) =>
                  List.generate(
                    3,
                    (c) => (p[c + 2] - mean[c]).abs(),
                  ).reduce(math.max) <
                  24,
            )
            .toList();
        field.add(
          List.generate(3, (c) {
            final plane = clean.length >= 6 ? _fit(clean, c) : null;
            if (plane == null || plane[1].abs() + plane[2].abs() > 512) {
              return mean[c];
            }
            return (plane[0] + plane[1] * u + plane[2] * v).clamp(
              mean[c] - 20,
              mean[c] + 20,
            );
          }),
        );
      }
    }
    return BackgroundSurface._(fit, field);
  }

  List<int> color(double u, double v) {
    if (local == null) {
      return [
        for (final c in channels)
          (c[0] + c[1] * u + c[2] * v).round().clamp(0, 255),
      ];
    }
    final x = (u * (gridSize - 1)).clamp(0, gridSize - 1 - 1e-9);
    final y = (v * (gridSize - 1)).clamp(0, gridSize - 1 - 1e-9);
    final ix = x.floor(), iy = y.floor(), fx = x - ix, fy = y - iy;
    return List.generate(
      3,
      (c) =>
          (local![iy * gridSize + ix][c] * (1 - fx) * (1 - fy) +
                  local![iy * gridSize + ix + 1][c] * fx * (1 - fy) +
                  local![(iy + 1) * gridSize + ix][c] * (1 - fx) * fy +
                  local![(iy + 1) * gridSize + ix + 1][c] * fx * fy)
              .round()
              .clamp(0, 255),
    );
  }

  static List<double>? _fit(List<List<double>> samples, int channel) {
    final a = List.generate(3, (_) => List.filled(4, 0.0));
    for (final s in samples) {
      final p = [1.0, s[0], s[1]];
      for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 3; j++) {
          a[i][j] += p[i] * p[j];
        }
        a[i][3] += p[i] * s[channel + 2];
      }
    }
    for (var col = 0; col < 3; col++) {
      var pivot = col;
      for (var row = col + 1; row < 3; row++) {
        if (a[row][col].abs() > a[pivot][col].abs()) pivot = row;
      }
      final swap = a[col];
      a[col] = a[pivot];
      a[pivot] = swap;
      if (a[col][col].abs() < 1e-8) return null;
      final divisor = a[col][col];
      for (var j = col; j < 4; j++) {
        a[col][j] /= divisor;
      }
      for (var row = 0; row < 3; row++) {
        if (row == col) continue;
        final factor = a[row][col];
        for (var j = col; j < 4; j++) {
          a[row][j] -= factor * a[col][j];
        }
      }
    }
    return [for (var i = 0; i < 3; i++) a[i][3]];
  }
}

/// Extend smooth stock colour from the actual repair boundary. Only missing
/// pixels are written; strokes and textured pixels cannot seed the extension.
void extendLocalStock(
  img.Image image,
  List<int> repaired,
  BackgroundSurface surface,
) {
  if (repaired.isEmpty) return;
  final w = image.width, h = image.height;
  final missing = List<bool>.filled(w * h, false);
  final distance = List<int>.filled(w * h, -1);
  for (final i in repaired) {
    missing[i] = true;
  }
  Iterable<int> neighbors(int i) sync* {
    final x = i % w, y = i ~/ w;
    if (x > 0) yield i - 1;
    if (x + 1 < w) yield i + 1;
    if (y > 0) yield i - w;
    if (y + 1 < h) yield i + w;
  }

  final donors = <int>{};
  for (final i in repaired) {
    for (final n in neighbors(i)) {
      if (!missing[n]) donors.add(n);
    }
  }
  final queue = <int>[];
  for (final i in donors) {
    final x = i % w, y = i ~/ w;
    final prior = surface.color((x + .5) / w, (y + .5) / h);
    final low = List<double>.filled(3, 255);
    final high = List<double>.filled(3, 0);
    final sum = List<double>.filled(3, 0);
    var count = 0;
    for (var yy = math.max(0, y - 2); yy <= math.min(h - 1, y + 2); yy++) {
      for (var xx = math.max(0, x - 2); xx <= math.min(w - 1, x + 2); xx++) {
        if (missing[yy * w + xx]) continue;
        final p = image.getPixel(xx, yy);
        if (p.a != 255) continue;
        final rgb = [p.r.toDouble(), p.g.toDouble(), p.b.toDouble()];
        for (var c = 0; c < 3; c++) {
          low[c] = math.min(low[c], rgb[c]);
          high[c] = math.max(high[c], rgb[c]);
          sum[c] += rgb[c];
        }
        count++;
      }
    }
    if (count < 6) continue;
    if (List.generate(3, (c) => high[c] - low[c]).reduce(math.max) > 18) {
      continue;
    }
    if (List.generate(
          3,
          (c) => (sum[c] / count - prior[c]).abs(),
        ).reduce(math.max) >
        32) {
      continue;
    }
    distance[i] = 0;
    queue.add(i);
  }
  // Breadth-first propagation averages only nearer, already accepted colours.
  // It cannot replicate a text stroke or modify a valid source pixel.
  for (var head = 0; head < queue.length; head++) {
    final i = queue[head];
    for (final n in neighbors(i)) {
      if (!missing[n] || distance[n] >= 0) continue;
      final d = distance[i] + 1;
      final sum = List<double>.filled(3, 0);
      var count = 0;
      for (final donor in neighbors(n)) {
        if (distance[donor] < 0 || distance[donor] >= d) continue;
        final p = image.getPixel(donor % w, donor ~/ w);
        sum[0] += p.r;
        sum[1] += p.g;
        sum[2] += p.b;
        count++;
      }
      image.setPixelRgba(
        n % w,
        n ~/ w,
        (sum[0] / count).round(),
        (sum[1] / count).round(),
        (sum[2] / count).round(),
        255,
      );
      distance[n] = d;
      queue.add(n);
    }
  }
  // Relax the propagated colours to avoid nearest-donor bands along an edge.
  // Only accepted smooth stock and missing pixels participate in this field.
  final values = <int, List<double>>{};
  for (final i in queue) {
    final p = image.getPixel(i % w, i ~/ w);
    values[i] = [p.r.toDouble(), p.g.toDouble(), p.b.toDouble()];
  }
  for (var iteration = 0; iteration < 60; iteration++) {
    var largest = 0.0;
    for (final i in repaired) {
      final current = values[i];
      if (current == null) continue;
      final sum = List<double>.filled(3, 0);
      var count = 0;
      for (final n in neighbors(i)) {
        final color = values[n];
        if (color == null) continue;
        for (var c = 0; c < 3; c++) {
          sum[c] += color[c];
        }
        count++;
      }
      if (count == 0) continue;
      for (var c = 0; c < 3; c++) {
        final change = 1.5 * (sum[c] / count - current[c]);
        current[c] = (current[c] + change).clamp(0, 255);
        largest = math.max(largest, change.abs());
      }
    }
    if (largest < .05) break;
  }
  for (final i in repaired) {
    final color = values[i];
    if (color != null) {
      image.setPixelRgba(
        i % w,
        i ~/ w,
        color[0].round(),
        color[1].round(),
        color[2].round(),
        255,
      );
    }
  }
}

/// Feather only a narrow, demonstrably smooth strip of existing background.
/// Detailed pixels never become donors, and are never blended.
void featherBackground(
  img.Image image,
  List<int> repaired,
  BackgroundSurface surface, {
  bool adjacentColor = false,
}) {
  if (repaired.isEmpty) return;
  final w = image.width, h = image.height;
  final radius = (math.min(w, h) / 100).round().clamp(3, 16);
  final distances = List<int>.filled(w * h, -1);
  final owners = List<int>.filled(w * h, -1);
  final queue = <int>[];
  for (final index in repaired) {
    distances[index] = 0;
    owners[index] = index;
    queue.add(index);
  }
  final original = img.Image.from(image);
  for (var head = 0; head < queue.length; head++) {
    final index = queue[head], d = distances[index];
    if (d >= radius) continue;
    final x = index % w, y = index ~/ w;
    for (final next in [
      if (x > 0) index - 1,
      if (x + 1 < w) index + 1,
      if (y > 0) index - w,
      if (y + 1 < h) index + w,
    ]) {
      if (distances[next] >= 0) continue;
      final nx = next % w, ny = next ~/ w;
      if (original.getPixel(nx, ny).a == 0) continue;
      distances[next] = d + 1;
      owners[next] = owners[index];
      queue.add(next);
    }
  }
  for (final index in queue) {
    final distance = distances[index];
    if (distance <= 0 || distance >= radius) continue;
    final x = index % w, y = index ~/ w;
    final donor = original.getPixel(owners[index] % w, owners[index] ~/ w);
    final base = adjacentColor
        ? [donor.r.toDouble(), donor.g.toDouble(), donor.b.toDouble()]
        : surface.color((x + .5) / w, (y + .5) / h);
    final p = original.getPixel(x, y);
    final color = [p.r.toDouble(), p.g.toDouble(), p.b.toDouble()];
    if (List.generate(3, (c) => (color[c] - base[c]).abs()).reduce(math.max) >
        12) {
      continue;
    }
    final low = List<double>.filled(3, 255), high = List<double>.filled(3, 0);
    var count = 0;
    // Inspect every pixel so a thin stroke cannot fall between sample points.
    for (
      var yy = math.max(0, y - radius);
      yy <= math.min(h - 1, y + radius);
      yy++
    ) {
      for (
        var xx = math.max(0, x - radius);
        xx <= math.min(w - 1, x + radius);
        xx++
      ) {
        if (distances[yy * w + xx] == 0) continue;
        final q = original.getPixel(xx, yy);
        if (q.a == 0) continue;
        final v = [q.r.toDouble(), q.g.toDouble(), q.b.toDouble()];
        for (var c = 0; c < 3; c++) {
          low[c] = math.min(low[c], v[c]);
          high[c] = math.max(high[c], v[c]);
        }
        count++;
      }
    }
    if (count < radius * radius ||
        List.generate(3, (c) => high[c] - low[c]).reduce(math.max) > 18) {
      continue;
    }
    final t = distance / radius;
    final weight = t * t * (3 - 2 * t);
    final blended = List.generate(
      3,
      (c) => (base[c] * (1 - weight) + color[c] * weight).round(),
    );
    image.setPixelRgba(x, y, blended[0], blended[1], blended[2], 255);
  }
}

/// Reject coherent exterior strips that were inside an over-wide detected edge.
/// A narrow stroke or an isolated dark glyph is not sufficient evidence.
List<int> exteriorStrips(img.Image image, BackgroundSurface surface) {
  final w = image.width, h = image.height;
  final depth = (math.min(w, h) * .025).round().clamp(4, 32);
  final result = <int>{};
  for (var side = 0; side < 4; side++) {
    final length = side.isEven ? w : h;
    int index(int t, int d) => switch (side) {
      0 => d * w + t,
      1 => t * w + w - 1 - d,
      2 => (h - 1 - d) * w + t,
      _ => t * w + d,
    };
    final cuts = List<int>.filled(length, 0);
    for (var t = 0; t < length; t++) {
      final first = index(t, 0);
      if (image.getPixel(first % w, first ~/ w).a == 0) continue;
      final outer = image.getPixel(first % w, first ~/ w);
      for (var d = 1; d < depth; d++) {
        final n = index(t, d);
        final p = image.getPixel(n % w, n ~/ w);
        final target = surface.color((n % w + .5) / w, (n ~/ w + .5) / h);
        final rgb = [p.r, p.g, p.b];
        final outside = [outer.r, outer.g, outer.b];
        final insideError = List.generate(
          3,
          (c) => (rgb[c] - target[c]).abs(),
        ).reduce(math.max);
        final outsideError = List.generate(
          3,
          (c) => (outside[c] - target[c]).abs(),
        ).reduce(math.max);
        if (p.a == 255 && insideError < 20 && outsideError > 55) {
          final before = index(t, d - 1);
          final q = image.getPixel(before % w, before ~/ w);
          if (math.max(
                (q.r - p.r).abs(),
                math.max((q.g - p.g).abs(), (q.b - p.b).abs()),
              ) >
              20) {
            cuts[t] = d;
            break;
          }
        }
      }
    }
    // Demand a long connected band, not a character-sized mark.
    var start = 0;
    while (start < length) {
      if (cuts[start] == 0) {
        start++;
        continue;
      }
      var end = start + 1;
      while (end < length &&
          cuts[end] > 0 &&
          (cuts[end] - cuts[end - 1]).abs() <= 2) {
        end++;
      }
      if (end - start >= math.max(24, (length * .18).round())) {
        for (var t = start; t < end; t++) {
          for (var d = 0; d < cuts[t]; d++) {
            result.add(index(t, d));
          }
        }
      }
      start = end;
    }
  }
  return result.toList();
}

/// Remove a dark fringe connected to an already missing edge. This is a flat
/// output repair mask, never a change to the photographed document boundary.
/// Isolated marks without missing-edge evidence are deliberately retained.
List<int> missingEdgeFringe(
  img.Image image,
  BackgroundSurface surface, {
  double depthFraction = .025,
  bool darkOnly = false,
}) {
  final w = image.width, h = image.height;
  final depth = (math.min(w, h) * depthFraction).round().clamp(
    4,
    depthFraction <= .025 ? 24 : 64,
  );
  final suspect = <int>{};
  for (var y = 0; y < h; y++) {
    for (var x = 0; x < w; x++) {
      if (math.min(math.min(x, w - 1 - x), math.min(y, h - 1 - y)) >= depth) {
        continue;
      }
      final p = image.getPixel(x, y);
      if (p.a == 0) continue;
      final base = surface.color((x + .5) / w, (y + .5) / h);
      final delta = math.max(
        (p.r - base[0]).abs(),
        math.max((p.g - base[1]).abs(), (p.b - base[2]).abs()),
      );
      final darkerThanStock =
          (base[0] + base[1] + base[2] - p.r - p.g - p.b) / 3 > 20;
      if (delta > 28 && (!darkOnly || darkerThanStock)) suspect.add(y * w + x);
    }
  }
  Iterable<int> adjacent(int i) sync* {
    final x = i % w, y = i ~/ w;
    if (x > 0) yield i - 1;
    if (x + 1 < w) yield i + 1;
    if (y > 0) yield i - w;
    if (y + 1 < h) yield i + w;
  }

  final result = <int>[];
  final visited = <int>{};
  for (final seed in suspect) {
    if (!visited.add(seed)) continue;
    final component = <int>[seed];
    var touchesMissing = false;
    final interior = <int>[];
    for (var head = 0; head < component.length; head++) {
      final i = component[head], x = i % w, y = i ~/ w;
      if (math.min(math.min(x, w - 1 - x), math.min(y, h - 1 - y)) >=
          depth - 1) {
        interior.add(i);
      }
      for (final next in adjacent(i)) {
        if (image.getPixel(next % w, next ~/ w).a == 0) touchesMissing = true;
        if (suspect.contains(next) && visited.add(next)) component.add(next);
      }
    }
    // Protect strokes continuing into content and their local neighbours.
    // One distant deep shadow must not veto a separate notch on this edge.
    if (!touchesMissing) continue;
    final protected = interior.toSet();
    var frontier = interior;
    for (var step = 0; step < depth * 2 && frontier.isNotEmpty; step++) {
      final nextLayer = <int>[];
      for (final i in frontier) {
        for (final next in adjacent(i)) {
          if (suspect.contains(next) && protected.add(next)) {
            nextLayer.add(next);
          }
        }
      }
      frontier = nextLayer;
    }
    result.addAll(component.where((i) => !protected.contains(i)));
  }
  return result;
}
