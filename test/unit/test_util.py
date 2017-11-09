import unittest
import sys
import numpy as np

sys.path.append('../../src/')

import util

class UtilTests(unittest.TestCase):
    def test_load_image(self):
        image_id = '6120_2_0'
        img = util.load_image(image_id, 'M')
        self.assertEqual(img.shape, (837, 851, 8))

    def test_normalize_image(self):
        image_id = '6120_2_0'
        img = util.load_image(image_id, 'M')
        normalized = util.normalize_image(img)
        self.assertTrue(np.max(normalized) <= 1)
        self.assertTrue(np.min(normalized) >= 0)
        self.assertEqual(img.shape, normalized.shape)

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
        height, width = img.shape[:2]
        shapes = util.load_wkt_shape(image_id)
        shape = shapes[0]
        xmax, ymin = util.load_grid_sizes(image_id)
        ext_coords, int_coords = util.convert_shape_to_coords(shape, height, width, xmax, ymin)
        self.assertEqual((573, 0), (len(ext_coords), len(int_coords)))
        for coords in ext_coords:
            self.assertTrue(np.all(coords >= 0))
            self.assertTrue(np.all(coords[:, 0] < height))
            self.assertTrue(np.all(coords[:, 1] < width))

    def test_create_mask(self):
        image_id = '6120_2_0'
        img = util.load_image(image_id, 'M')
        height, width = img.shape[:2]
        masks = util.create_mask(image_id, height, width)
        self.assertEqual((837, 851, 10), masks.shape)


if __name__ == '__main__':
    unittest.main()

