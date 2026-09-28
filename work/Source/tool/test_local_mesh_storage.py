import unittest
import tempfile
import json
from pathlib import Path
import local_mesh_storage as storage


class LocalMeshStorageTests(unittest.TestCase):
    def test_stores_original_cells_without_refitting_or_touching_parent(self):
        parent = {'rows': 5, 'columns': 5, 'grid': [[x / 4, y / 4] for y in range(5) for x in range(5)],
                  'method': 'accepted parent'}
        original = {**parent, 'preserved_mesh': parent, 'refinement_regions': [[.25, .25, .75, .75]]}
        result = storage.compact(original)
        self.assertEqual(result['grid_window'], [5, 5, 1, 1])
        self.assertEqual((result['rows'], result['columns']), (3, 3))
        self.assertEqual(result['grid'], [original['grid'][y * 5 + x]
                                         for y in range(1, 4) for x in range(1, 4)])
        self.assertIs(result['preserved_mesh'], parent)
        self.assertNotIn('grid_window', original)
        self.assertIs(storage.compact(result), result)

    def test_root_and_unsupported_refinements_are_not_reformatted(self):
        base = {'method': 'accepted root'}
        self.assertIs(storage.compact(base), base)

    def test_failed_publication_keeps_the_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'mesh.json'
            accepted = {'method': 'accepted', 'value': .25}
            storage.publish(accepted, path)
            with self.assertRaises(ValueError):
                storage.publish({'value': float('nan')}, path)
            self.assertEqual(json.loads(path.read_text()), accepted)


if __name__ == '__main__':
    unittest.main()
