import unittest
import sys

sys.path.append('../../src/')

import util

class UtilTests(unittest.TestCase):
    def test_load_image(self):
        image_id = '6120_2_0'
        img = util.load_image(image_id, 'M')
        self.assertEqual(tuple(img.shape), (837, 851, 8))

    def test_load_grid_sizes(self):
        image_id = '6120_2_0'
        xmax, ymin = util.load_grid_sizes(image_id)
        self.assertAlmostEqual(xmax, 0.009188)
        self.assertAlmostEqual(ymin, -0.00904)

    def test_load_wkt_shape(self):
        image_id = '6120_2_0'
        shapes = util.load_wkt_shape(image_id)
        self.assertEqual(len(shapes), util.num_class)
        classes = list(range(5))
        shapes = util.load_wkt_shape(image_id, classes)
        self.assertEqual(len(shapes), 5)

    def test_convert_shape_to_coords(self):
        image_id = '6120_2_0'
        img = util.load_image(image_id, 'M')
        shapes = util.load_wkt_shape(image_id)
        shape = shapes[0]
        height, width = img.shape[:2]
        xmax, ymin = util.load_grid_sizes(image_id)
        ext_coords, int_coords = util.convert_shape_to_coords(shape, height, width, xmax, ymin)


if __name__ == '__main__':
    unittest.main()

