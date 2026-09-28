"""Store only the grid nodes used by a bounded refinement, without refitting."""
import math
import json
from pathlib import Path


def compact(result, *, recursive=False):
    """Crop whole existing cells; the original global sample indices stay exact.

    Older accepted parents remain untouched by default. Recursive compaction is
    opt-in so previously approved layers are never reformatted incidentally.
    """
    parent = result.get('preserved_mesh')
    regions = result.get('refinement_regions', [])
    if 'grid_window' in result and parent is not None and recursive:
        return {**result, 'preserved_mesh': compact(parent, recursive=True)}
    if parent is None or not regions or 'grid_window' in result:
        return result
    rows, columns = result['rows'], result['columns']
    left = max(0, math.floor(min(r[0] for r in regions) * (columns - 1)))
    top = max(0, math.floor(min(r[1] for r in regions) * (rows - 1)))
    right = min(columns - 1, math.ceil(max(r[2] for r in regions) * (columns - 1)))
    bottom = min(rows - 1, math.ceil(max(r[3] for r in regions) * (rows - 1)))
    if left >= right or top >= bottom:
        raise ValueError('Empty refinement window')
    grid = result['grid']
    return {
        **result,
        'rows': bottom - top + 1,
        'columns': right - left + 1,
        'grid': [point for y in range(top, bottom + 1)
                 for point in grid[y * columns + left:y * columns + right + 1]],
        'grid_window': [rows, columns, top, left],
        'preserved_mesh': compact(parent, recursive=True) if recursive else parent,
    }


def publish(result, output):
    """Atomically replace only a completed checkpoint, retaining prior on error."""
    output = Path(output)
    temporary = output.with_name('.' + output.name + '.partial')
    encoded = json.dumps(compact(result), allow_nan=False)
    temporary.write_text(encoded)
    temporary.replace(output)
