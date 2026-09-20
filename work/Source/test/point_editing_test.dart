import 'dart:convert';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pen_tool/main.dart';

void main() {
  Store sample() => Store()..points = [
    PenPoint(x: .2, y: .2), PenPoint(x: .8, y: .2),
    PenPoint(x: .8, y: .8), PenPoint(x: .2, y: .8),
  ];

  test('Reveal switches engine off without changing reference geometry', () {
    final s = sample()..showPrediction = true..toolMode = true;
    final before = jsonEncode(s.points.map((p) => p.toJson()).toList());
    s.toggleReferenceReveal();
    expect(s.showPrediction, false);
    expect(s.toolMode, true);
    expect(jsonEncode(s.points.map((p) => p.toJson()).toList()), before);
    expect(s.dirty, false);
    s.toggleReferenceReveal();
    expect(s.showPrediction, false);
    expect(s.toolMode, false);
  });

  test('box selection and handle reveal do not alter document geometry', () {
    final s = sample();
    final original = jsonEncode(s.points.map((p) => p.toJson()).toList());
    s.selectArea(const Rect.fromLTRB(.1, .1, .9, .3));
    expect(s.multiSelect, {0, 1});
    s.revealSelection();
    expect(s.revealed, {0, 1});
    expect(s.dirty, false);
    expect(jsonEncode(s.points.map((p) => p.toJson()).toList()), original);
  });

  test('handle movement changes curve while anchor stays fixed and undo restores it', () {
    final s = sample()..multiSelect = {0};
    s.revealSelection();
    s.pushHistory();
    s.moveHandle(0, 'hOut', .03, -.02);
    final p = s.points.first;
    expect(p.x, .2); expect(p.y, .2); expect(p.smooth, true);
    expect(p.hIn[0] + p.hOut[0], closeTo(.4, 1e-9));
    expect(p.hIn[1] + p.hOut[1], closeTo(.4, 1e-9));
    s.undo();
    expect(s.points.first.smooth, false);
    s.redo();
    expect(s.points.first.smooth, true);
  });

  test('independent handle changes do not move opposite handle', () {
    final s = sample()..multiSelect = {0};
    s.revealSelection();
    final opposite = [...s.handleAt(0, 'hIn')];
    s.pushHistory();
    s.moveHandle(0, 'hOut', .03, -.02, independent: true);
    expect(s.points.first.hIn, opposite);
    expect(s.points.first.broken, true);
    s.undo();
    expect(s.points.first.broken, false);
  });
}
