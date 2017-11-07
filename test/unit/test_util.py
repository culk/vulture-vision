import unittest
import sys

sys.path.append('../../src/')

import util

class UtilTests(unittest.TestCase):
    def test_load_wkt_shapes(self):
        image_id = '6120_2_0'
        shapes = util.load_wkt_shape(image_id)
        self.assertEqual(len(shapes), util.num_class)
        classes = list(range(5))
        shapes = util.load_wkt_shape(image_id, classes)
        self.assertEqual(len(shapes), 5)


if __name__ == '__main__':
    unittest.main()

