"""Add shared word/rule row alignment after the previously accepted steps."""
import sys

import cv2
import numpy as np

import frame_section_candidate as frame
import section_row_candidate as rows
from local_mesh_storage import publish
from printed_rule_sections import measure_sections, word_ink_mask
from word_rule_membership import separate_rule_ink


def refine(path, base, *, audit_log=None, word_ink_groups=None, separate_structural_ink=False,
           exact_ordering=False, independent_word_owners=False, balanced_target=False,
           stock_relaxation=0, minimum_resolution=(4097,1025),
           stock_coordinate_relaxation=0,precise_geometry=False,words_mask=None,
           preserve_neighbouring_rules=False):
    gray, safe, _, _ = rows.current_measurement(path, base)
    h, w = gray.shape
    words = word_ink_mask(gray) if words_mask is None else np.asarray(words_mask,dtype=np.uint8)
    horizontal = measure_sections(gray, safe, word_mask=words)
    vertical = measure_sections(gray.T, safe.T, word_mask=words.T)
    # A shared printed form grid supplies evidence for this next stage. The
    # established receipt and isolated-word paths remain the base correction.
    if (sum(np.ptp(g[:, 0]) > .3 for g in horizontal) < 3 or
            sum(np.ptp(g[:, 0]) > .2 for g in vertical) < 3):
        return base
    # Component summaries alone missed faint letter edges in a real form.
    # Do not accept a new row without independent whole-word ink evidence.
    if not word_ink_groups:
        if audit_log is not None:
            audit_log.append({'rejected': 'whole-word evidence unavailable'})
        return base
    groups = rows.associate_rows(horizontal, w, h)
    groups.sort(key=lambda group: np.ptp(np.percentile(np.vstack(group)[:, 1], [5, 95])) *
                sum(np.ptp(g[:, 0]) for g in group), reverse=True)
    for group in groups:
        clean_words = word_ink_groups
        if word_ink_groups:
            clean_words, _ = separate_rule_ink(word_ink_groups, np.vstack(group), w, h)
        target_guide=None
        if balanced_target:
            points=np.vstack(group)
            center=float(points[:,1].mean())
            target_guide=np.array([[points[:,0].min(),center],[points[:,0].max(),center]])
        result = rows.refine(path, base, guides=group, align_to_existing=True,
                             target_guide=target_guide,
                             words_mask=words,
                             include_wide_ink=True, minimum_resolution=minimum_resolution,
                             audit_log=audit_log, word_ink_groups=clean_words,
                             separate_structural_ink=separate_structural_ink,
                             exact_ordering=exact_ordering,
                             independent_word_owners=independent_word_owners,
                             stock_relaxation=stock_relaxation,
                             stock_coordinate_relaxation=stock_coordinate_relaxation,
                             precise_geometry=precise_geometry,
                             preserve_neighbouring_rules=preserve_neighbouring_rules)
        if result is not base:
            result['method'] = 'Experimental shared word and rule row correction'
            return result
    return base


def solve(path, working=None):
    base = frame.solve(path, working)
    try:
        return refine(path, base)
    except (OSError, ValueError, ArithmeticError, AttributeError, IndexError,
            KeyError, TypeError, cv2.error):
        return base


def solve_staged(path, output, working=None):
    base = frame.solve_staged(path, output, working)
    try:
        result = refine(path, base)
        if result is not base:
            publish(result, output)
        return result
    except (OSError, ValueError, ArithmeticError, AttributeError, IndexError,
            KeyError, TypeError, cv2.error):
        return base


if __name__ == '__main__':
    solve_staged(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
