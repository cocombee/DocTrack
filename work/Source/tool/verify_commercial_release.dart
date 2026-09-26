import 'dart:convert';
import 'dart:io';

Never _fail(String message) {
  stderr.writeln('Commercial clearance failed: $message');
  exit(1);
}

void main(List<String> arguments) {
  final source = Directory.current;
  final app = Directory(
    arguments.isEmpty
        ? '${source.parent.path}/New Build/DocTrack.app'
        : arguments.single,
  );
  if (!app.existsSync()) _fail('app bundle does not exist: ${app.path}');

  final assets = Directory(
    '${app.path}/Contents/Frameworks/App.framework/Versions/A/Resources/'
    'flutter_assets',
  );
  final generatedNotices = File('${assets.path}/NOTICES.Z');
  final legalNotices = File(
    '${assets.path}/assets/legal/document_scanning_notices.txt',
  );
  final manifestFile = File(
    '${assets.path}/assets/legal/document_scanning_manifest.json',
  );
  for (final required in [generatedNotices, legalNotices, manifestFile]) {
    if (!required.existsSync() || required.lengthSync() == 0) {
      _fail('missing legal resource: ${required.path}');
    }
  }

  final forbiddenExtensions = {'.onnx', '.pkl', '.pt', '.pth', '.tflite'};
  final forbiddenNames = ['fairscan', 'docres', 'lama'];
  final forbidden = <String>[];
  for (final entity in app.listSync(recursive: true, followLinks: false)) {
    if (entity is! File) continue;
    final lower = entity.path.toLowerCase();
    if (forbiddenExtensions.any(lower.endsWith) ||
        forbiddenNames.any(lower.contains)) {
      forbidden.add(entity.path);
    }
  }
  if (forbidden.isNotEmpty) {
    _fail('unapproved files are bundled:\n${forbidden.join('\n')}');
  }

  final manifest = jsonDecode(manifestFile.readAsStringSync());
  if (manifest is! Map<String, dynamic>) _fail('invalid manifest root');
  final components = manifest['components'];
  if (components is! List) _fail('manifest has no component list');
  for (final value in components) {
    if (value is! Map<String, dynamic>) _fail('invalid component entry');
    if (value['bundled'] == true && value['commercialApproved'] != true) {
      _fail('bundled component is not approved: ${value['name']}');
    }
  }

  late final String registry;
  try {
    registry = utf8.decode(gzip.decode(generatedNotices.readAsBytesSync()));
  } catch (_) {
    _fail('Flutter licence registry is unreadable');
  }
  for (final name in ['image', 'cupertino_icons', 'onnxruntime']) {
    if (!registry.toLowerCase().contains(name)) {
      _fail('$name notice is absent from Flutter licence registry');
    }
  }

  stdout.writeln(
    'Commercial clearance passed: approved notices are present and no '
    'unapproved model weights or rejected evaluation assets are bundled.',
  );
}
