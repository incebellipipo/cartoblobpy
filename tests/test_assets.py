import os
import unittest

from cartoblobpy.assets import (
    get_map_path,
    list_example_maps,
    save_result,
    load_result,
    list_available_results,
)


class TestAssets(unittest.TestCase):
    def test_list_and_paths_exist(self):
        maps = list_example_maps()
        self.assertIsInstance(maps, list)
        self.assertTrue(any(m.endswith(".yaml") for m in maps))
        for m in maps:
            self.assertTrue(os.path.isfile(get_map_path(m)))

    def test_save_and_load_result(self):
        payload = {"a": 1, "b": [1, 2, 3]}
        fname = "unittest_payload.pkl"
        path = save_result(payload, fname)
        self.assertTrue(os.path.isfile(path))

        loaded = load_result(fname)
        self.assertEqual(loaded, payload)

        # Cleanup
        os.remove(path)
        # list_available_results should not include the removed file
        self.assertNotIn(fname, list_available_results())


if __name__ == "__main__":
    unittest.main()
