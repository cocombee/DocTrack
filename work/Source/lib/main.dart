import 'package:tracker_engine/tracker_engine.dart';

import 'scanner/review_scanner.dart';
import 'scanner/detected_text_cache.dart';
import 'flattening/flatten_preview.dart';
import 'flattening/flatten_image.dart';
import 'flattening/isolate_document.dart';
import 'flattening/processing_loading.dart';
import 'flattening/document_words.dart';

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/gestures.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'document_plane.dart';
import 'doctrack_logo.dart';
import 'curve_reveal_icon.dart';

void main() => runApp(const PenToolApp());

const root =
    '/Users/mrkokom/My Drive (cocombees@gmail.com)/Codex/Apps/DocTrack/work/Data';

// shadcn dark (zinc) tokens
const _bg = Color(0xFF09090B);
const _card = Color(0xFF111113);
const _border = Color(0xFF27272A);
const _secondary = Color(0xFF1C1C21);
const _fg = Color(0xFFFAFAFA);
const _muted = Color(0xFFA1A1AA);

// ---------- data ----------

class PenPoint {
  double x, y;
  bool corner, smooth;
  bool broken; // true after Alt-drag: handles move independently (no mirror)
  List<double> hIn, hOut;
  PenPoint({
    required this.x,
    required this.y,
    this.corner = false,
    this.smooth = false,
    List<double>? hIn,
    List<double>? hOut,
    this.broken = false,
  }) : hIn = hIn ?? [x - 0.05, y],
       hOut = hOut ?? [x + 0.05, y];

  Map<String, dynamic> toJson() => {
    'x': _r5(x),
    'y': _r5(y),
    'corner': corner,
    'smooth': smooth,
    'broken': broken,
    'hIn': [_r4(hIn[0]), _r4(hIn[1])],
    'hOut': [_r4(hOut[0]), _r4(hOut[1])],
  };

  static double _r5(double v) => (v * 100000).roundToDouble() / 100000;
  static double _r4(double v) => (v * 10000).roundToDouble() / 10000;

  factory PenPoint.fromJson(dynamic q) {
    if (q is Map) {
      final x = (q['x'] as num?)?.toDouble() ?? 0.0;
      final y = (q['y'] as num?)?.toDouble() ?? 0.0;
      List<double> arr(dynamic v, double def) => v is List && v.length >= 2
          ? [v[0].toDouble(), v[1].toDouble()]
          : [def, def];
      return PenPoint(
        x: x,
        y: y,
        corner: q['corner'] == true,
        smooth: q['smooth'] == true,
        broken: q['broken'] == true,
        hIn: arr(q['hIn'], x - 0.05),
        hOut: arr(q['hOut'], x + 0.05),
      );
    }
    return PenPoint(x: 0, y: 0);
  }

  PenPoint clone() => PenPoint(
    x: x,
    y: y,
    corner: corner,
    smooth: smooth,
    broken: broken,
    hIn: [...hIn],
    hOut: [...hOut],
  );
}

class LabelFile {
  String image, sourceId, notes;
  List<List<double>> corners;
  List<String> order, conditions, occlusion;
  List<PenPoint>? fullPolygon;
  bool uncertain;
  double? insetPx;
  String? edgeRule, cornerStyle, variantOf;

  LabelFile({
    required this.image,
    required this.sourceId,
    required this.corners,
    this.fullPolygon,
    this.uncertain = false,
    this.conditions = const [],
    this.notes = 'outer paper edge',
    this.occlusion = const [],
    this.order = const ['top-left', 'top-right', 'bottom-right', 'bottom-left'],
    this.insetPx,
    this.edgeRule,
    this.cornerStyle,
    this.variantOf,
  });

  factory LabelFile.fromJson(Map<String, dynamic> j) => LabelFile(
    image: j['image'] ?? '',
    sourceId: j['source_id'] ?? 'doc',
    corners:
        (j['corners'] as List?)
            ?.map((c) => <double>[(c as List)[0].toDouble(), c[1].toDouble()])
            .toList() ??
        [],
    fullPolygon: (j['fullPolygon'] as List?)?.map(PenPoint.fromJson).toList(),
    uncertain: j['uncertain'] == true,
    conditions:
        (j['conditions'] as List?)?.map((e) => e.toString()).toList() ?? [],
    notes: j['notes'] ?? '',
    occlusion:
        (j['occlusion'] as List?)?.map((e) => e.toString()).toList() ?? [],
    order:
        (j['order'] as List?)?.map((e) => e.toString()).toList() ??
        const ['top-left', 'top-right', 'bottom-right', 'bottom-left'],
    insetPx: (j['inset_px'] as num?)?.toDouble(),
    edgeRule: j['edge_rule'],
    cornerStyle: j['corner_style'],
    variantOf: j['variant_of'],
  );

  Map<String, dynamic> toJson() => {
    'image': image,
    'source_id': sourceId,
    'variant_of': variantOf,
    'corners': corners,
    'order': order,
    'fullPolygon': fullPolygon?.map((p) => p.toJson()).toList(),
    'uncertain': uncertain,
    'occlusion': occlusion,
    'conditions': conditions,
    'notes': notes,
    if (insetPx != null) 'inset_px': insetPx,
    if (edgeRule != null) 'edge_rule': edgeRule,
    if (cornerStyle != null) 'corner_style': cornerStyle,
  };
}

class Sample {
  final int id;
  final String split, imagePath, labelPath, name;
  Sample({
    required this.id,
    required this.split,
    required this.imagePath,
    required this.labelPath,
    required this.name,
  });
}

// ---------- store ----------

class Store extends ChangeNotifier {
  Store({this.detector});
  final Future<BoundaryResult> Function(Uint8List, ReviewScanCancellation)?
  detector;
  ReviewScanCancellation? _scanCancellation;
  void cancelScanAndReveal() {
    detectionRevision++;
    _scanCancellation?.cancel();
    _scanCancellation = null;
    detecting = false;
    showPrediction = false;
    detectionError = null;
    toolMode = true;
    revealSelection();
    refresh();
  }

  void toggleEngineScan() {
    if (detecting || showPrediction) {
      cancelScanAndReveal();
    } else {
      detectCurrent();
    }
  }

  List<Sample> samples = [];
  int index = 0;
  List<PenPoint> points = [];
  LabelFile? meta;
  bool dirty = false, loaded = false;
  bool toolMode = false; // curve tool: click a dot reveals its curve instantly
  Set<int> multiSelect = {};
  final Set<int> revealed = {};
  final Map<int, List<List<double>>> handlePreview = {};
  String status = 'loading…';
  double zoom = 1;
  BoundaryResult? prediction;
  bool detecting = false, showPrediction = false;
  final Map<String, BoundaryResult> _predictionCache = {};
  final DetectedTextCache _predictionTextCache = DetectedTextCache();
  List<String>? predictionTextLines(String imagePath, Uint8List bytes) =>
      prediction != null && current?.imagePath == imagePath
      ? _predictionTextCache.lookup(imagePath, bytes)
      : null;
  List<DocumentWordGeometry>? predictionPhotoWords(
    String imagePath,
    Uint8List bytes,
  ) => prediction != null && current?.imagePath == imagePath
      ? _predictionTextCache.lookupWords(imagePath, bytes)
      : null;
  int detectionRevision = 0;
  String? detectionError;
  Future<void> detectCurrent({bool reveal = true}) async {
    final sample = current;
    if (sample == null || detecting) return;
    final revision = ++detectionRevision;
    final cancellation = ReviewScanCancellation();
    _scanCancellation = cancellation;
    detecting = true;
    detectionError = null;
    prediction = null;
    if (reveal) showPrediction = true;
    refresh();
    try {
      final bytes = await File('$root/${sample.imagePath}').readAsBytes();
      cancellation.check();
      List<String>? detectedLines;
      List<DocumentWordGeometry>? detectedWords;
      final result =
          await (detector?.call(bytes, cancellation) ??
              detectReviewImage(
                bytes,
                cancellation: cancellation,
                onTextLines: (lines) => detectedLines = lines,
                onPhotoWords: (words) => detectedWords = words,
              ));
      if (revision != detectionRevision) return;
      prediction = result;
      _predictionCache[sample.imagePath] = result;
      _predictionTextCache.record(
        sample.imagePath,
        bytes,
        detectedLines,
        words: detectedWords,
      );
    } catch (error) {
      if (revision == detectionRevision) {
        detectionError = 'Detection failed: $error';
      }
    } finally {
      if (revision == detectionRevision) {
        detecting = false;
        refresh();
      }
    }
  }

  // dot-drag interaction state (shared with canvas to suppress ghost dots)
  Timer? holdTimer;
  bool dragMoved = false;
  bool holding = false;
  DateTime? lastDotPanAt;
  int? selectedIndex;

  final Map<int, List<List<PenPoint>>> _hist = {};
  final Map<int, List<List<PenPoint>>> _redo = {};
  final Map<int, int> _savedDots = {};

  void refresh() => notifyListeners();

  void clearSelection() {
    selectedIndex = null;
    multiSelect.clear();
    revealed.clear();
    handlePreview.clear();
    notifyListeners();
  }

  void toggleReferenceReveal() {
    if (detecting) {
      cancelScanAndReveal();
      return;
    }
    // Reveal always leaves the engine layer before exposing reference handles.
    toolMode = showPrediction || !toolMode;
    showPrediction = false;
    if (toolMode) {
      revealSelection();
    } else {
      revealed.clear();
    }
    refresh();
  }

  void revealSelection() {
    final indices = {...multiSelect, ?selectedIndex};
    for (final i in indices) {
      if (i < 0 || i >= points.length) continue;
      revealed.add(i);
      final p = points[i];
      if (p.smooth || handlePreview.containsKey(i)) continue;
      final prev = points[(i - 1 + points.length) % points.length];
      final next = points[(i + 1) % points.length];
      final a = Offset(p.x - prev.x, p.y - prev.y);
      final b = Offset(next.x - p.x, next.y - p.y);
      var tangent =
          a / math.max(a.distance, 1e-9) + b / math.max(b.distance, 1e-9);
      if (tangent.distance < .2) {
        tangent = a.distance > 1e-9 ? a : const Offset(1, 0);
      }
      tangent /= tangent.distance;
      final length = math.max(math.min(a.distance, b.distance) * .175, .02);
      handlePreview[i] = [
        [p.x - tangent.dx * length, p.y - tangent.dy * length],
        [p.x + tangent.dx * length, p.y + tangent.dy * length],
      ];
    }
    notifyListeners();
  }

  List<double> handleAt(int i, String side) {
    final p = points[i];
    final preview = !p.smooth ? handlePreview[i] : null;
    return preview?[side == 'hIn' ? 0 : 1] ?? (side == 'hIn' ? p.hIn : p.hOut);
  }

  void moveHandle(
    int i,
    String side,
    double dx,
    double dy, {
    bool independent = false,
  }) {
    final p = points[i];
    if (!p.smooth) {
      p.hIn = [...handleAt(i, 'hIn')];
      p.hOut = [...handleAt(i, 'hOut')];
      p.smooth = true;
      p.corner = false;
      handlePreview.remove(i);
    }
    if (independent) p.broken = true;
    final h = side == 'hIn' ? p.hIn : p.hOut;
    h[0] += dx;
    h[1] += dy;
    if (!p.broken) {
      final other = side == 'hIn' ? p.hOut : p.hIn;
      other[0] = 2 * p.x - h[0];
      other[1] = 2 * p.y - h[1];
    }
    dirty = true;
    notifyListeners();
  }

  void selectArea(Rect area) {
    multiSelect = {
      for (var i = 0; i < points.length; i++)
        if (area.contains(Offset(points[i].x, points[i].y))) i,
    };
    selectedIndex = multiSelect.isEmpty ? null : multiSelect.last;
    revealed.retainAll(multiSelect);
    if (toolMode) revealSelection();
    notifyListeners();
  }

  Sample? get current =>
      samples.isNotEmpty && index < samples.length ? samples[index] : null;

  Future<void> loadManifest() async {
    try {
      final arr =
          jsonDecode(await File('$root/manifest.json').readAsString()) as List;
      samples = arr
          .asMap()
          .entries
          .map(
            (e) => Sample(
              id: e.key,
              split: e.value['split'] ?? '',
              imagePath: e.value['image'] ?? '',
              labelPath: e.value['label'] ?? '',
              name: e.value['name'] ?? '',
            ),
          )
          .toList();
      status = samples.isEmpty ? 'no samples' : '${samples.length} samples';
    } catch (e) {
      status = 'manifest error: $e';
    }
    notifyListeners();
  }

  Future<void> loadImage(int idx) async {
    if (idx < 0 || idx >= samples.length) return;
    _scanCancellation?.cancel();
    _scanCancellation = null;
    detectionRevision++;
    detecting = false;
    prediction = _predictionCache[samples[idx].imagePath];
    detectionError = null;
    index = idx;
    points = [];
    meta = null;
    loaded = false;
    zoom = 1;
    selectedIndex = null;
    holding = false;
    dragMoved = false;
    multiSelect = {};
    revealed.clear();
    handlePreview.clear();
    holdTimer?.cancel();
    notifyListeners();
    final revision = detectionRevision;
    await readLabel(samples[idx]);
    if (revision == detectionRevision && showPrediction && prediction == null) {
      await detectCurrent(reveal: false);
    }
  }

  Future<void> readLabel(Sample s) async {
    final revision = detectionRevision;
    final path = '$root/${s.labelPath}';
    try {
      final f = File(path);
      if (!f.existsSync()) {
        meta = null;
        points = [];
        loaded = true;
        dirty = false;
        status = 'fresh — no label';
        notifyListeners();
        return;
      }
      final L = LabelFile.fromJson(jsonDecode(await f.readAsString()));
      if (revision != detectionRevision || current?.id != s.id) return;
      // NEVER downgrade a dense polygon to 4 corners
      if (L.fullPolygon != null && L.fullPolygon!.isNotEmpty) {
        points = L.fullPolygon!;
      } else if (L.corners.length == 4) {
        points = L.corners
            .map(
              (c) => PenPoint(
                x: c[0],
                y: c[1],
                hIn: [c[0] - 0.05, c[1]],
                hOut: [c[0] + 0.05, c[1]],
              ),
            )
            .toList();
      }
      meta = L;
      loaded = true;
      dirty = false;
      _savedDots[s.id] = points.length;
      status = '${points.length} dots';
    } catch (e) {
      if (revision != detectionRevision || current?.id != s.id) return;
      loaded = true;
      status = 'LABEL PARSE FAILED: ${s.name} — $e';
    }
    notifyListeners();
  }

  Future<void> reloadCurrent() async {
    final s = current;
    if (s == null) return;
    _hist.remove(index);
    _redo.remove(index);
    dirty = false;
    await readLabel(s);
  }

  void pushHistory() {
    final h = _hist.putIfAbsent(index, () => []);
    h.add(points.map((p) => p.clone()).toList());
    if (h.length > 100) h.removeAt(0);
    _redo[index] = [];
    dirty = true;
  }

  void undo() {
    final h = _hist[index];
    if (h == null || h.isEmpty) return;
    (_redo[index] ??= []).add(points.map((p) => p.clone()).toList());
    points = h.removeLast();
    revealed.clear();
    handlePreview.clear();
    dirty = true;
    notifyListeners();
  }

  void redo() {
    final r = _redo[index];
    if (r == null || r.isEmpty) return;
    (_hist[index] ??= []).add(points.map((p) => p.clone()).toList());
    points = r.removeLast();
    revealed.clear();
    handlePreview.clear();
    dirty = true;
    notifyListeners();
  }

  String? save() {
    final s = current;
    if (s == null) return 'no image';
    if (points.length < 4) return 'need ≥4 dots';
    final tl = _ext(points, (p) => -(p.x + p.y));
    final tr = _ext(points, (p) => p.x - p.y);
    final br = _ext(points, (p) => p.x + p.y);
    final bl = _ext(points, (p) => -(p.x - p.y));
    final L =
        LabelFile(
            image: '${s.split}/${s.name}',
            sourceId: s.name.split('-').first,
            corners: [
              tl,
              tr,
              br,
              bl,
            ].map((p) => [PenPoint._r5(p.x), PenPoint._r5(p.y)]).toList(),
            fullPolygon: points,
            uncertain: meta?.uncertain ?? false,
            conditions: meta?.conditions ?? [],
            notes: meta?.notes.isNotEmpty == true
                ? meta!.notes
                : 'outer paper edge',
            occlusion: meta?.occlusion ?? [],
            order:
                meta?.order ??
                const ['top-left', 'top-right', 'bottom-right', 'bottom-left'],
          )
          ..insetPx = meta?.insetPx
          ..edgeRule = meta?.edgeRule
          ..cornerStyle = meta?.cornerStyle
          ..variantOf = meta?.variantOf;
    try {
      File('$root/${s.labelPath}').writeAsStringSync(
        const JsonEncoder.withIndent('  ').convert(L.toJson()),
      );
      dirty = false;
      _savedDots[s.id] = points.length;
      status = 'saved ✓';
      notifyListeners();
      return null;
    } catch (e) {
      return 'save failed: $e';
    }
  }

  PenPoint _ext(List<PenPoint> pts, double Function(PenPoint) fn) =>
      pts.reduce((a, b) => fn(b) > fn(a) ? b : a);
}

// ---------- app ----------

class PenToolApp extends StatelessWidget {
  const PenToolApp({super.key});
  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'DocTrack',
    debugShowCheckedModeBanner: false,
    theme: ThemeData.dark(useMaterial3: true),
    home: const _Home(),
  );
}

class _Home extends StatefulWidget {
  const _Home();
  @override
  State<_Home> createState() => _HomeState();
}

class _HomeState extends State<_Home> {
  final store = Store();
  final Map<String, _FlattenCacheEntry> _flattenCache = {};
  bool flattenPreviewOpen = false;
  FlattenPreview? flatPreview;
  final flatKey = GlobalKey<FlattenPreviewState>();
  String? flatImagePath;
  int flatRevision = 0;
  String? flatError;
  DateTime? flatStartedAt;
  String flatStage = 'Finding the document…';
  Completer<void>? flatDetectionWait;

  void _closeFlatten() {
    if (!mounted) return;
    if (flatDetectionWait?.isCompleted == false) flatDetectionWait!.complete();
    flatDetectionWait = null;
    setState(() {
      flatRevision++;
      flatError = null;
      flattenPreviewOpen = false;
      flatPreview = null;
      flatImagePath = null;
    });
  }

  void _flattenNavigationChanged() {
    if (flattenPreviewOpen &&
        store.current != null &&
        store.current!.imagePath != flatImagePath) {
      // Navigation is allowed to swap the cached preview immediately. The
      // preview widget then decides whether a render is needed; do not start a
      // second render just because the selected sample changed.
      unawaited(_prepareFlatten());
    }
  }

  Future<void> _flatten() async {
    if (flattenPreviewOpen) {
      _closeFlatten();
      return;
    }
    if (store.current == null) return;
    await _prepareFlatten();
  }

  Future<void> _prepareFlatten() async {
    if (flatDetectionWait?.isCompleted == false) flatDetectionWait!.complete();
    flatDetectionWait = null;
    final sample = store.current!;
    final request = ++flatRevision;
    setState(() {
      flattenPreviewOpen = true;
      flatImagePath = sample.imagePath;
      flatError = null;
      flatPreview = null;
      flatStartedAt = DateTime.now();
      flatStage = 'Finding the document…';
    });
    bool current() =>
        mounted &&
        request == flatRevision &&
        store.current?.imagePath == sample.imagePath;
    try {
      if (store.detecting) {
        final wait = Completer<void>();
        flatDetectionWait = wait;
        void finished() {
          if ((!store.detecting || !current()) && !wait.isCompleted) {
            wait.complete();
          }
        }

        store.addListener(finished);
        try {
          finished();
          await wait.future;
        } finally {
          store.removeListener(finished);
          if (identical(flatDetectionWait, wait)) flatDetectionWait = null;
        }
      }
      if (!current()) return;
      if (store.prediction == null) await store.detectCurrent();
      if (!current()) return;
      final boundary = store.prediction;
      if (boundary == null || boundary.corners?.length != 4) {
        throw StateError("Detect a four-corner document first.");
      }
      final cacheKey = _flattenBoundaryKey(boundary);
      final bytes = await File("$root/${sample.imagePath}").readAsBytes();
      if (!current()) return;
      // Preview caching is disabled while mesh mapping is being corrected.
      // Every flatten request reruns isolation, mesh detection, and repair.
      if (mounted && current()) {
        setState(() => flatPreview = null);
      }
      var textLines = store.predictionTextLines(sample.imagePath, bytes);
      setState(() => flatStage = 'Reading the document…');
      if (textLines == null) {
        try {
          textLines =
              await const MethodChannel('doctrack/window')
                  .invokeListMethod<String>('recognizeDocumentText', {
                    'data': bytes,
                    'id': 0,
                  }) ??
              [];
        } on PlatformException {
          /* Geometry remains available when OCR fails. */
        } on MissingPluginException {
          /* Other hosts can supply format evidence. */
        }
      }
      final resolvedTextLines = textLines ?? <String>[];
      if (!current()) return;
      setState(
        () => flatStage = 'Separating the document from its background…',
      );
      final isolated = await isolateDocument(
        bytes,
        boundary,
        textLines: resolvedTextLines,
      );
      if (!current()) return;
      setState(
        () => flatPreview = FlattenPreview(
          key: flatKey,
          processingStartedAt: flatStartedAt,
          onClose: _closeFlatten,
          bytes: isolated.png,
          measurementBytes: isolated.measurementPng,
          boundary: isolated.boundary,
          name: sample.name,
          textLines: resolvedTextLines,
          estimatedRatio: isolated.estimatedRatio,
          outputDirectory: "$root/../Flattened Outputs",
          onResult: (result) {
            _flattenCache[sample.imagePath] = _FlattenCacheEntry(
              boundaryKey: cacheKey,
              originalBytes: bytes,
              boundary: isolated.boundary,
              sourceBytes: isolated.png,
              measurementBytes: isolated.measurementPng,
              textLines: List<String>.from(resolvedTextLines),
              estimatedRatio: isolated.estimatedRatio,
              result: result,
            );
          },
        ),
      );
    } catch (e) {
      if (!mounted || !current()) return;
      setState(() => flatError = 'Could not flatten: $e');
    }
  }

  String _flattenBoundaryKey(BoundaryResult boundary) => [
    'workflow-v3-stable-preview',
    boundary.width,
    boundary.height,
    ...?boundary.corners?.expand((p) => [p.x, p.y]),
    jsonEncode(boundary.boundarySegments.map((s) => s.toJson()).toList()),
  ].join(':');

  @override
  void initState() {
    super.initState();
    FocusManager.instance.addEarlyKeyEventHandler(_navigationKey);
    store.addListener(_flattenNavigationChanged);
    store.loadManifest().then((_) => store.loadImage(0));
  }

  KeyEventResult _navigationKey(KeyEvent event) {
    final keyboard = HardwareKeyboard.instance;
    if (keyboard.isControlPressed ||
        keyboard.isMetaPressed ||
        keyboard.isAltPressed ||
        keyboard.isShiftPressed) {
      return KeyEventResult.ignored;
    }
    final key = event.logicalKey;
    final target = key == LogicalKeyboardKey.arrowLeft
        ? store.index - 1
        : key == LogicalKeyboardKey.arrowRight
        ? store.index + 1
        : key == LogicalKeyboardKey.arrowUp
        ? 0
        : key == LogicalKeyboardKey.arrowDown
        ? store.samples.length - 1
        : null;
    if (target == null) return KeyEventResult.ignored;
    if (event is KeyDownEvent || event is KeyRepeatEvent) {
      store.loadImage(target);
    }
    return KeyEventResult.handled;
  }

  @override
  void dispose() {
    FocusManager.instance.removeEarlyKeyEventHandler(_navigationKey);
    store.removeListener(_flattenNavigationChanged);
    if (flatDetectionWait?.isCompleted == false) flatDetectionWait!.complete();
    flatDetectionWait = null;
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: store,
      builder: (context, _) => CallbackShortcuts(
        bindings: {
          const SingleActivator(LogicalKeyboardKey.keyZ, meta: true):
              store.undo,
          const SingleActivator(
            LogicalKeyboardKey.keyZ,
            meta: true,
            shift: true,
          ): store.redo,
          const SingleActivator(LogicalKeyboardKey.keyS, meta: true): _save,
          const SingleActivator(LogicalKeyboardKey.bracketLeft): () =>
              store.loadImage(store.index - 1),
          const SingleActivator(LogicalKeyboardKey.bracketRight): () =>
              store.loadImage(store.index + 1),
          const SingleActivator(LogicalKeyboardKey.delete): _deleteSelected,
          const SingleActivator(LogicalKeyboardKey.backspace): _deleteSelected,
          const SingleActivator(LogicalKeyboardKey.escape): () =>
              store.clearSelection(),
        },
        child: Focus(
          autofocus: true,
          child: Scaffold(
            backgroundColor: _bg,
            body: Row(
              children: [
                _sidebar(),
                const VerticalDivider(width: 1, color: _border),
                Expanded(child: _canvasArea()),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _save() {
    if (flattenPreviewOpen) {
      flatKey.currentState?.saveOutput();
      return;
    }
    final err = store.save();
    if (err != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(err), duration: const Duration(seconds: 2)),
      );
    }
  }

  void _deleteSelected() {
    final indices =
        {
            ...store.multiSelect,
            ?store.selectedIndex,
          }.where((i) => i >= 0 && i < store.points.length).toList()
          ..sort((a, b) => b.compareTo(a));
    if (indices.isEmpty || store.points.length - indices.length < 4) return;
    store.pushHistory();
    for (final i in indices) {
      store.points.removeAt(i);
    }
    store.clearSelection();
  }

  Widget _toolBtn({
    required IconData icon,
    Widget? glyph,
    required bool active,
    required String tip,
    required VoidCallback onTap,
  }) {
    return Tooltip(
      message: tip,
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        child: GestureDetector(
          onTap: onTap,
          child: Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: active ? const Color(0xFFFAFAFA) : _secondary,
              border: Border.all(
                color: active ? const Color(0xFFFAFAFA) : _border,
              ),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Center(
              child: SizedBox.square(
                dimension: 18,
                child:
                    glyph ??
                    Icon(
                      icon,
                      size: 18,
                      color: active ? const Color(0xFF09090B) : _fg,
                    ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _windowControl(
    String label,
    String action,
    Color color,
    IconData icon,
  ) => Tooltip(
    message: label,
    child: Semantics(
      button: true,
      label: label,
      child: MouseRegion(
        cursor: SystemMouseCursors.click,
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTap: () =>
              const MethodChannel('doctrack/window').invokeMethod<void>(action),
          child: SizedBox(
            width: 22,
            height: 22,
            child: Center(
              child: Container(
                width: 13,
                height: 13,
                decoration: BoxDecoration(color: color, shape: BoxShape.circle),
                child: Icon(icon, size: 9, color: const Color(0x99000000)),
              ),
            ),
          ),
        ),
      ),
    ),
  );

  Widget _sidebar() => SizedBox(
    width: 262,
    child: Container(
      color: _card,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Padding(
            padding: const EdgeInsets.fromLTRB(12, 8, 12, 0),
            child: Row(
              children: [
                _windowControl(
                  'Close window',
                  'close',
                  const Color(0xFFFF5F57),
                  Icons.close,
                ),
                _windowControl(
                  'Minimize window',
                  'minimize',
                  const Color(0xFFFFBD2E),
                  Icons.remove,
                ),
                _windowControl(
                  'Maximize or restore window',
                  'maximize',
                  const Color(0xFF28C840),
                  Icons.add,
                ),
              ],
            ),
          ),
          const Padding(
            padding: EdgeInsets.fromLTRB(16, 12, 16, 20),
            child: Row(
              children: [
                DocTrackLogo(size: 32),
                SizedBox(width: 10),
                Text(
                  'DocTrack',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    color: _fg,
                  ),
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Text(
              '${store.status}\nDrag to select points · Curve shows handles\nClick edge to add · Right-click to delete',
              style: const TextStyle(color: _muted, fontSize: 11, height: 1.5),
            ),
          ),
          const Divider(color: _border, height: 1),
          Expanded(
            child: ListView.builder(
              itemCount: store.samples.length,
              itemBuilder: (_, i) {
                final sel = i == store.index;
                final dotCount = i == store.index
                    ? store.points.length
                    : store._savedDots[store.samples[i].id] ?? 0;
                return Material(
                  color: sel ? _secondary : Colors.transparent,
                  child: InkWell(
                    onTap: () => store.loadImage(i),
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 16,
                        vertical: 10,
                      ),
                      decoration: const BoxDecoration(
                        border: Border(
                          bottom: BorderSide(color: Color(0xFF1F1F23)),
                        ),
                      ),
                      child: Row(
                        children: [
                          Expanded(
                            child: Text(
                              '${i + 1}. ${store.samples[i].name}',
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontSize: 12.5,
                                fontWeight: FontWeight.w400,
                                color: _fg,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(
                            dotCount > 0 ? '✓ $dotCount' : '·',
                            style: TextStyle(
                              fontSize: 12.5,
                              color: dotCount > 0
                                  ? const Color(0xFF4ADE80)
                                  : Colors.white38,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    ),
  );

  Widget _canvasArea() => Column(
    children: [
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        decoration: const BoxDecoration(
          border: Border(bottom: BorderSide(color: _border)),
          color: _bg,
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(
              width: double.infinity,
              child: Tooltip(
                message: store.current?.name ?? '',
                child: Text(
                  '${store.index + 1}/${store.samples.length} — ${store.current?.name ?? '—'}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontWeight: FontWeight.w400,
                    color: _fg,
                    fontSize: 13,
                  ),
                ),
              ),
            ),
            const SizedBox(height: 6),
            Text(
              '${store.current?.split ?? ''}  ·  ${store.dirty ? 'unsaved •' : 'saved ✓'}  ·  ${(store.zoom * 100).round()}%',
              style: const TextStyle(color: _muted, fontSize: 11),
            ),
          ],
        ),
      ),
      Expanded(
        child: Container(
          color: Colors.black,
          child: LayoutBuilder(
            builder: (context, box) {
              final s = store.current;
              if (s == null || !File('$root/${s.imagePath}').existsSync()) {
                return const Center(
                  child: Text(
                    'no image',
                    style: TextStyle(color: Colors.white24),
                  ),
                );
              }
              return Stack(
                children: [
                  Positioned.fill(
                    child: flattenPreviewOpen
                        ? flatPreview ??
                              (flatError != null
                                  ? Center(child: Text(flatError!))
                                  : ImageProcessingLoading(
                                      startedAt: flatStartedAt!,
                                      stage: flatStage,
                                      onClose: _closeFlatten,
                                    ))
                        : _ZoomableCanvas(
                            key: ValueKey(s.imagePath),
                            store: store,
                          ),
                  ),
                  // Floating tools: curve toggle on top, reset below (shadcn square)
                  Positioned(
                    right: 12,
                    top: 12,
                    child: Column(
                      children: [
                        _toolBtn(
                          icon: Icons.timeline,
                          glyph: CurveRevealIcon(
                            color: store.toolMode ? _bg : _fg,
                          ),
                          active: store.toolMode,
                          tip: 'Reveal reference and its curve handles',
                          onTap: () {
                            _closeFlatten();
                            store.toggleReferenceReveal();
                          },
                        ),
                        const SizedBox(height: 8),
                        _toolBtn(
                          icon: Icons.refresh,
                          active: false,
                          tip: 'Reload label from disk',
                          onTap: () {
                            _closeFlatten();
                            store.reloadCurrent();
                          },
                        ),
                        const SizedBox(height: 8),
                        _toolBtn(
                          icon: store.detecting
                              ? Icons.hourglass_top
                              : Icons.document_scanner_outlined,
                          active: store.showPrediction,
                          tip: store.detecting || store.showPrediction
                              ? 'Cancel engine scan and reveal reference'
                              : 'Detect document automatically',
                          onTap: () {
                            _closeFlatten();
                            store.toggleEngineScan();
                          },
                        ),
                        const SizedBox(height: 8),
                        _toolBtn(
                          icon: Icons.auto_awesome_mosaic_outlined,
                          active: flattenPreviewOpen,
                          tip: 'Flatten document',
                          onTap: _flatten,
                        ),
                      ],
                    ),
                  ),
                  Positioned(
                    right: 12,
                    bottom: 12,
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        _toolBtn(
                          icon: Icons.save_outlined,
                          active: false,
                          tip: flattenPreviewOpen
                              ? 'Save flattened PNG'
                              : 'Save',
                          onTap: _save,
                        ),
                        const SizedBox(width: 8),
                        _toolBtn(
                          icon: Icons.arrow_back,
                          active: false,
                          tip: 'Previous image',
                          onTap: () => store.loadImage(store.index - 1),
                        ),
                        const SizedBox(width: 8),
                        _toolBtn(
                          icon: Icons.arrow_forward,
                          active: false,
                          tip: 'Next image',
                          onTap: () => store.loadImage(store.index + 1),
                        ),
                      ],
                    ),
                  ),
                ],
              );
            },
          ),
        ),
      ),
    ],
  );
}

class _FlattenCacheEntry {
  const _FlattenCacheEntry({
    required this.boundaryKey,
    required this.originalBytes,
    required this.boundary,
    required this.sourceBytes,
    required this.measurementBytes,
    required this.textLines,
    required this.estimatedRatio,
    required this.result,
  });
  final String boundaryKey;
  final Uint8List originalBytes;
  final BoundaryResult boundary;
  final Uint8List sourceBytes;
  final Uint8List? measurementBytes;
  final List<String> textLines;
  final double? estimatedRatio;
  final FlattenedImage result;
}

// ---------- canvas ----------
// Image + tracker live in ONE InteractiveViewer: zoom moves both together,
// so the tracker can never detach from the paper.

class _ZoomableCanvas extends StatefulWidget {
  final Store store;
  const _ZoomableCanvas({super.key, required this.store});
  @override
  State<_ZoomableCanvas> createState() => _ZoomableCanvasState();
}

class _ZoomableCanvasState extends State<_ZoomableCanvas> {
  int? dragIndex;
  String? handleSide;
  Offset? lastPos;
  Offset? _midPanStart;
  Matrix4? _midPanMatrix;
  double? aspect;
  final TransformationController _tc = TransformationController();
  String? _loadedPath;
  int? _loadedIndex;

  Store get store => widget.store;

  @override
  void initState() {
    super.initState();
    _tc.addListener(_syncZoom);
  }

  void _syncZoom() {
    final store = widget.store;
    final z = _tc.value.getMaxScaleOnAxis();
    if ((z - store.zoom).abs() > 0.001) {
      store.zoom = z;
      store.refresh();
    }
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final s = widget.store.current;
    if (s != null &&
        (s.imagePath != _loadedPath || widget.store.index != _loadedIndex)) {
      _loadedPath = s.imagePath;
      _loadedIndex = widget.store.index;
      _loadAspect(s.imagePath);
      _tc.value = Matrix4.identity();
    }
  }

  Future<void> _loadAspect(String path) async {
    try {
      final bytes = File('$root/$path').readAsBytesSync();
      final d = await ui.instantiateImageCodec(bytes);
      final frame = await d.getNextFrame();
      if (!mounted) return;
      setState(() => aspect = frame.image.width / frame.image.height);
    } catch (_) {}
  }

  Size _fit(BoxConstraints b) {
    final a = aspect ?? 1.0;
    var w = b.maxWidth, h = w / a;
    if (h > b.maxHeight) {
      h = b.maxHeight;
      w = h * a;
    }
    return Size(w, h);
  }

  @override
  Widget build(BuildContext context) {
    final store = widget.store;
    final s = store.current;
    if (s == null) return const SizedBox.shrink();
    return LayoutBuilder(
      builder: (context, box) {
        final fit = _fit(box);
        return Listener(
          onPointerDown: (d) {
            if (d.buttons == kMiddleMouseButton) {
              _midPanStart = d.position;
              _midPanMatrix = _tc.value.clone();
            }
          },
          onPointerMove: (d) {
            if (_midPanStart != null && d.buttons == kMiddleMouseButton) {
              final delta = d.position - _midPanStart!;
              _tc.value = _midPanMatrix!.clone()
                ..translateByDouble(delta.dx, delta.dy, 0, 1);
            }
          },
          onPointerUp: (d) {
            if (d.buttons != kMiddleMouseButton) _midPanStart = null;
          },
          onPointerCancel: (_) => _midPanStart = null,
          child: InteractiveViewer(
            transformationController: _tc,
            minScale: 0.3,
            maxScale: 6,
            panEnabled: false,
            trackpadScrollCausesScale: true,
            child: DocumentPlane(
              size: fit,
              children: [
                Positioned.fill(
                  child: Image.file(
                    File('$root/${s.imagePath}'),
                    fit: BoxFit.fill,
                    gaplessPlayback: false,
                  ),
                ),
                Positioned.fill(
                  child: MouseRegion(
                    cursor: SystemMouseCursors.basic,
                    child: GestureDetector(
                      behavior: HitTestBehavior.translucent,
                      onPanDown: (d) => _canvasDown(d.localPosition, fit),
                      onTapUp: (d) => _tap(d.localPosition, fit),
                      onPanStart: (d) => _createDragStart(d.localPosition, fit),
                      onPanUpdate: (d) =>
                          _createDragUpdate(d.localPosition, fit),
                      onPanEnd: (_) => _createDragEnd(),
                      child: CustomPaint(
                        size: fit,
                        painter: _OverlayPainter(
                          store,
                          fit,
                          marquee: _marqueeStart == null
                              ? null
                              : Rect.fromPoints(_marqueeStart!, _marqueeEnd!),
                        ),
                      ),
                    ),
                  ),
                ),
                Positioned.fill(
                  child: Stack(
                    children: [
                      if (!store.showPrediction)
                        for (var i = 0; i < store.points.length; i++)
                          _dot(i, fit),
                      if (!store.showPrediction)
                        for (final i in store.revealed)
                          if (i >= 0 && i < store.points.length)
                            for (final side in ['hIn', 'hOut'])
                              _handle(i, side, fit),
                    ],
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  // ---- Marquee drag-select (drag on empty canvas) ----
  Offset? _marqueeStart;
  Offset? _marqueeEnd;

  void _marqueeStartDrag(Offset pos, Size fit) {
    _marqueeStart = pos;
    _marqueeEnd = pos;
    widget.store.refresh();
  }

  void _marqueeUpdate(Offset pos, Size fit) {
    _marqueeEnd = pos;
    final store = widget.store;
    final tl = Offset(
      math.min(_marqueeStart!.dx, _marqueeEnd!.dx),
      math.min(_marqueeStart!.dy, _marqueeEnd!.dy),
    );
    final br = Offset(
      math.max(_marqueeStart!.dx, _marqueeEnd!.dx),
      math.max(_marqueeStart!.dy, _marqueeEnd!.dy),
    );
    store.selectArea(
      Rect.fromLTRB(
        tl.dx / fit.width,
        tl.dy / fit.height,
        br.dx / fit.width,
        br.dy / fit.height,
      ),
    );
  }

  void _marqueeEndDrag() {
    _marqueeStart = null;
    _marqueeEnd = null;
    widget.store.refresh();
  }

  // ---- Affinity-style creation: click-drag pulls tangent handles out live ----
  Offset? _downPos;
  int? _creatingIndex;

  void _canvasDown(Offset pos, Size fit) {
    if (widget.store.showPrediction) return;
    _downPos = pos;
    _creatingIndex = null;
  }

  void _createDragStart(Offset pos, Size fit) {
    if (widget.store.showPrediction) return;
    final store = widget.store;
    if (_downPos == null) return;
    final last = store.lastDotPanAt;
    if (last != null && DateTime.now().difference(last).inMilliseconds < 450) {
      return;
    }
    final grabPx = 14.0 / store.zoom;
    for (final p in store.points) {
      if ((Offset(p.x * fit.width, p.y * fit.height) - _downPos!).distance <=
          grabPx) {
        return;
      }
    }
    _marqueeStartDrag(_downPos!, fit);
    _marqueeUpdate(pos, fit);
  }

  void _createDragUpdate(Offset pos, Size fit) {
    final store = widget.store;
    if (_marqueeStart != null) {
      _marqueeUpdate(pos, fit);
      return;
    }
    final i = _creatingIndex;
    if (i == null || i >= store.points.length) return;
    final p = store.points[i];
    final hx = (pos.dx / fit.width).clamp(0.0, 1.0).toDouble();
    final hy = (pos.dy / fit.height).clamp(0.0, 1.0).toDouble();
    p.hOut = [hx, hy];
    p.hIn = [(2 * p.x - hx).clamp(0.0, 1.0), (2 * p.y - hy).clamp(0.0, 1.0)];
    store.refresh();
  }

  void _createDragEnd() {
    final store = widget.store;
    if (_marqueeStart != null) {
      _marqueeEndDrag();
      return;
    }
    final i = _creatingIndex;
    _creatingIndex = null;
    _downPos = null;
    if (i == null || i >= store.points.length) return;
    final p = store.points[i];
    final handleLen = math.sqrt(
      (p.hOut[0] - p.x) * (p.hOut[0] - p.x) +
          (p.hOut[1] - p.y) * (p.hOut[1] - p.y),
    );
    if (handleLen < 0.004) {
      // no meaningful pull = plain corner point
      p.smooth = false;
      p.hIn = [p.x - 0.05, p.y];
      p.hOut = [p.x + 0.05, p.y];
    }
    store.refresh();
  }

  Widget _dot(int i, Size fit) {
    final store = widget.store;
    return Positioned(
      left: store.points[i].x * fit.width - 7 / store.zoom,
      top: store.points[i].y * fit.height - 7 / store.zoom,
      child: MouseRegion(
        cursor: SystemMouseCursors.grab,
        child: Listener(
          onPointerDown: (_) {
            // Hold 0.7s WITHOUT moving => spawn curve control handles.
            store.dragMoved = false;
            store.holding = false;
            store.holdTimer?.cancel();
            store.holdTimer = Timer(const Duration(milliseconds: 700), () {
              if (!store.dragMoved) {
                store.revealSelection();
                store.holding = true;
              }
            });
          },
          onPointerUp: (_) => store.holdTimer?.cancel(),
          onPointerCancel: (_) => store.holdTimer?.cancel(),
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTapDown: (_) {
              setState(() {
                store.selectedIndex = i;
                if (HardwareKeyboard.instance.isShiftPressed) {
                  if (!store.multiSelect.add(i)) store.multiSelect.remove(i);
                } else if (!store.multiSelect.contains(i)) {
                  store.multiSelect = {i};
                  store.revealed.clear();
                }
                if (store.toolMode) store.revealSelection();
              });
            },
            onPanStart: (d) {
              dragIndex = i;
              handleSide = null;
              lastPos = d.globalPosition;
            },
            onPanUpdate: (d) =>
                _panUpdate(d.globalPosition, fit, i, handleSide: null),
            onPanEnd: (_) {
              store.holdTimer?.cancel();
              store.lastDotPanAt = DateTime.now();
              dragIndex = null;
              store.holding = false;
              store.dragMoved = false;
            },
            onSecondaryTapUp: (_) => _deletePoint(i),
            child: Container(
              width: 14 / store.zoom,
              height: 14 / store.zoom,
              color: Colors.transparent,
            ),
          ),
        ),
      ),
    );
  }

  Widget _handle(int i, String side, Size fit) {
    final store = widget.store;
    return Positioned(
      left: store.handleAt(i, side)[0] * fit.width - 9 / store.zoom,
      top: store.handleAt(i, side)[1] * fit.height - 9 / store.zoom,
      child: MouseRegion(
        cursor: SystemMouseCursors.move,
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onPanStart: (d) {
            dragIndex = i;
            handleSide = side;
            lastPos = d.globalPosition;
            store.dragMoved = false;
          },
          onPanUpdate: (d) =>
              _panUpdate(d.globalPosition, fit, i, handleSide: side),
          onPanEnd: (_) {
            dragIndex = null;
            store.dragMoved = false;
          },
          child: Container(
            width: 18 / store.zoom,
            height: 18 / store.zoom,
            color: Colors.transparent,
          ),
        ),
      ),
    );
  }

  void _tap(Offset pos, Size fit) {
    if (widget.store.showPrediction) return;
    final store = widget.store;
    // A dot drag/release must never spawn a dot: suppress canvas taps
    // shortly after any dot pan interaction.
    final last = store.lastDotPanAt;
    if (last != null && DateTime.now().difference(last).inMilliseconds < 450) {
      return;
    }
    if (store.selectedIndex != null || store.multiSelect.isNotEmpty) {
      store.clearSelection();
      return;
    }
    // Ignore taps on/near an existing dot.
    final grabPx = 14.0 / store.zoom;
    for (final p in store.points) {
      if ((Offset(p.x * fit.width, p.y * fit.height) - pos).distance <=
          grabPx) {
        return;
      }
    }
    if (_creatingIndex != null) return;
    final x = (pos.dx / fit.width).clamp(0.0, 1.0).toDouble();
    final y = (pos.dy / fit.height).clamp(0.0, 1.0).toDouble();
    final np = PenPoint(x: x, y: y, hIn: [x - 0.05, y], hOut: [x + 0.05, y]);
    store.pushHistory();
    if (store.points.length >= 3) {
      // click on/near a line = insert into that segment (perimeter order kept)
      int bi = 0;
      double bd = double.infinity;
      for (var i = 0; i < store.points.length; i++) {
        final a = store.points[i],
            b = store.points[(i + 1) % store.points.length];
        final vx = b.x - a.x, vy = b.y - a.y;
        final l2 = vx * vx + vy * vy + 1e-12;
        var t = ((x - a.x) * vx + (y - a.y) * vy) / l2;
        t = t.clamp(0.0, 1.0);
        final d = math.sqrt(
          (x - a.x - t * vx) * (x - a.x - t * vx) +
              (y - a.y - t * vy) * (y - a.y - t * vy),
        );
        if (d < bd) {
          bd = d;
          bi = i;
        }
      }
      store.points.insert(bi + 1, np);
    } else {
      store.points.add(np);
    }
    store.dirty = true;
    store.refresh();
  }

  void _deletePoint(int i) {
    final store = widget.store;
    if (store.points.length <= 4) return;
    store.pushHistory();
    store.points.removeAt(i);
    store.clearSelection();
    store.dirty = true;
    store.refresh();
  }

  void _panUpdate(Offset pos, Size fit, int i, {String? handleSide}) {
    final store = widget.store;
    if (lastPos == null) {
      lastPos = pos;
      return;
    }
    final dx = (pos.dx - lastPos!.dx) / (fit.width * store.zoom);
    final dy = (pos.dy - lastPos!.dy) / (fit.height * store.zoom);
    if (dx == 0 && dy == 0) return;
    if (!store.dragMoved) {
      store.dragMoved = true;
      store.holdTimer?.cancel();
      store.pushHistory();
    }
    lastPos = pos;
    if (handleSide != null) {
      store.moveHandle(
        i,
        handleSide,
        dx,
        dy,
        independent: HardwareKeyboard.instance.isAltPressed,
      );
    } else {
      final selected = store.multiSelect.contains(i) ? store.multiSelect : {i};
      var mx = dx, my = dy;
      for (final j in selected) {
        mx = mx.clamp(-store.points[j].x, 1 - store.points[j].x);
        my = my.clamp(-store.points[j].y, 1 - store.points[j].y);
      }
      for (final j in selected) {
        final p = store.points[j];
        p.x += mx;
        p.y += my;
        for (final h in [p.hIn, p.hOut, ...?store.handlePreview[j]]) {
          h[0] += mx;
          h[1] += my;
        }
      }
      store.dirty = true;
      store.refresh();
    }
  }
}

class _OverlayPainter extends CustomPainter {
  final Store store;
  final Size fit;
  final Rect? marquee;
  _OverlayPainter(this.store, this.fit, {this.marquee});

  @override
  void paint(Canvas canvas, Size size) {
    final pts = store.points;
    final screenUnit = 1.0 / store.zoom;
    if (store.showPrediction && store.prediction == null) return;
    if (store.showPrediction && store.prediction != null) {
      final prediction = store.prediction!;
      final segments = offsetBoundarySegments(
        prediction.boundarySegments,
        prediction.width,
        prediction.height,
        4,
      );
      if (segments.isEmpty) return;
      final shape = Path()
        ..moveTo(
          segments.first.start.x * fit.width,
          segments.first.start.y * fit.height,
        );
      for (final segment in segments) {
        if (segment.isCurve) {
          shape.cubicTo(
            segment.control1!.x * fit.width,
            segment.control1!.y * fit.height,
            segment.control2!.x * fit.width,
            segment.control2!.y * fit.height,
            segment.end.x * fit.width,
            segment.end.y * fit.height,
          );
        } else {
          shape.lineTo(segment.end.x * fit.width, segment.end.y * fit.height);
        }
      }
      shape.close();
      canvas.drawPath(shape, Paint()..color = const Color(0x2240C4FF));
      canvas.drawPath(
        shape,
        Paint()
          ..color = const Color(0xFF40C4FF)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2 * screenUnit,
      );
      for (final segment in segments) {
        Offset position(TrackerPoint p) =>
            Offset(p.x * fit.width, p.y * fit.height);
        if (store.toolMode && segment.isCurve) {
          final handles = Paint()
            ..color = const Color(0xFF40C4FF)
            ..strokeWidth = screenUnit;
          canvas.drawLine(
            position(segment.start),
            position(segment.control1!),
            handles,
          );
          canvas.drawLine(
            position(segment.end),
            position(segment.control2!),
            handles,
          );
          for (final point in [segment.control1!, segment.control2!]) {
            canvas.drawRect(
              Rect.fromCenter(
                center: position(point),
                width: 6 * screenUnit,
                height: 6 * screenUnit,
              ),
              handles,
            );
          }
        }
        canvas.drawCircle(
          position(segment.start),
          4 * screenUnit,
          Paint()..color = const Color(0xFFFFB74D),
        );
      }
      return;
    }
    if (marquee != null) {
      canvas.drawRect(marquee!, Paint()..color = const Color(0x2238BDF8));
      canvas.drawRect(
        marquee!,
        Paint()
          ..color = const Color(0xFF38BDF8)
          ..style = PaintingStyle.stroke
          ..strokeWidth = 1 * screenUnit,
      );
    }
    if (pts.length < 3) return;
    final W = fit.width, H = fit.height;
    Offset P(PenPoint p) => Offset(p.x * W, p.y * H);

    final shape = Path()..moveTo(pts[0].x * W, pts[0].y * H);
    final n = pts.length;
    for (var i = 0; i < n; i++) {
      final a = pts[i], b = pts[(i + 1) % n];
      // Corner nodes are their own control point (true straight edges);
      // smooth nodes bend their adjacent segment via hOut/hIn.
      final c1 = a.smooth ? Offset(a.hOut[0] * W, a.hOut[1] * H) : P(a);
      final c2 = b.smooth ? Offset(b.hIn[0] * W, b.hIn[1] * H) : P(b);
      if (a.smooth || b.smooth) {
        shape.cubicTo(c1.dx, c1.dy, c2.dx, c2.dy, P(b).dx, P(b).dy);
      } else {
        shape.lineTo(P(b).dx, P(b).dy);
      }
    }
    shape.close();

    canvas.drawPath(shape, Paint()..color = const Color(0x2E4ADE80));
    canvas.drawPath(
      shape,
      Paint()
        ..color = const Color(0xFF16A34A)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2 * screenUnit,
    );

    for (var i = 0; i < n; i++) {
      final p = pts[i];
      final sel = i == store.selectedIndex;
      // multi-selected nodes get a subtle green ring
      if (store.multiSelect.contains(i) && !sel) {
        canvas.drawCircle(
          P(p),
          8 * screenUnit,
          Paint()
            ..color = const Color(0xFF4ADE80)
            ..style = PaintingStyle.stroke
            ..strokeWidth = 1.5 * screenUnit,
        );
      }
      // Affinity convention: handles visible ONLY for the selected node AND
      // only when it is a real curve node (smooth). Corner nodes have no
      // handles — placeholder stubs must never be drawn or draggable.
      if (store.revealed.contains(i)) {
        for (final h in [store.handleAt(i, 'hIn'), store.handleAt(i, 'hOut')]) {
          canvas.drawLine(
            P(p),
            Offset(h[0] * W, h[1] * H),
            Paint()
              ..color = const Color(0xFF60A5FA)
              ..strokeWidth = 1 * screenUnit,
          );
          canvas.drawCircle(
            Offset(h[0] * W, h[1] * H),
            5 * screenUnit,
            Paint()..color = const Color(0xFF60A5FA),
          );
        }
      }
      final nodePaint = Paint()
        ..color = sel ? const Color(0xFFEF4444) : Colors.white;
      if (p.smooth) {
        // smooth node = circle
        canvas.drawCircle(P(p), 5 * screenUnit, nodePaint);
      } else {
        // corner node = square (Affinity convention)
        canvas.drawRect(
          Rect.fromCenter(
            center: P(p),
            width: 9 * screenUnit,
            height: 9 * screenUnit,
          ),
          nodePaint,
        );
      }
      if (sel) {
        canvas.drawCircle(
          P(p),
          8 * screenUnit,
          Paint()
            ..color = Colors.white
            ..style = PaintingStyle.stroke
            ..strokeWidth = 2 * screenUnit,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant _OverlayPainter old) => true;
}
