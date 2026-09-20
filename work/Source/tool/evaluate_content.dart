import 'dart:io';
import 'dart:convert';
import 'package:tracker_engine/tracker_engine.dart';
void main(){
 final source=jsonDecode(File('/tmp/doctrack-ocr-lines.json').readAsStringSync()) as Map;
 final manifest=jsonDecode(File('../Data/manifest.json').readAsStringSync()) as List;
 for(var i=0;i<manifest.length;i++){
  final e=DocumentContentEvidence.fromLines((source[manifest[i]['name']] as List).cast<String>());
  stdout.writeln('${i+1}: ${e.kind.name} transaction=${e.transactionTerms} identity=${e.identityTerms} amounts=${e.amountRows}');
 }
}
