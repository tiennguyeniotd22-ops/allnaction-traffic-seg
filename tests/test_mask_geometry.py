"""Run python -m unittest discover -s tests from the repository root."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.mask_geometry import export_coordinates, polygon_area_at_640
from scripts.validate_and_augment import validate_split


class MaskAreaTest(unittest.TestCase):
    def test_exact_threshold_each_orientation(self):
        for w,h in [(640,640),(1280,720),(720,1280)]:
            scale = max(w,h)/640
            for area in (63,64,65):
                with self.subTest(width=w, height=h, area=area):
                    p = np.array([[10,10],[10+area*scale,10],
                                  [10+area*scale,10+scale],[10,10+scale]])/[w,h]
                    _, exported = export_coordinates(p)
                    self.assertEqual(polygon_area_at_640(exported,w,h), area)

    def test_without_rounding(self):
        # 143 source pixels are 63.555... square pixels at 640.
        p = np.array([[0,0],[13,0],[13,11],[0,11]])/[960,600]
        self.assertAlmostEqual(polygon_area_at_640(p,960,600), 143*(2/3)**2)
        self.assertLess(polygon_area_at_640(p,960,600), 64)

    def test_validator_counts_errors_and_backgrounds(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            imgs, labels = root/"images/train", root/"labels/train"
            imgs.mkdir(parents=True)
            labels.mkdir(parents=True)
            for name in ("labeled","background","missing"):
                Image.new("RGB",(1280,720)).save(imgs/f"{name}.png")
            (labels/"background.txt").touch()
            (labels/"labeled.txt").write_text("\n".join([
                "2.9 0 0 1 0 0 1", "11 0 0 1 0 0 1", "0 nan 0 1 0 0 1",
                "0 0 0 inf 0 0 1", "0 0 0 a 0 0 1", "0 0 0 1 0 0 1.1",
                "0 0 0 1 0", "1 0 0 .01 0 .01 .01 0 .01",
                "2 0 0 .1 0 0 .1"]))
            r = validate_split(root,"train")
            self.assertEqual(r["bad_lines"], 7)
            self.assertEqual(sum(r["class_counts"].values()), 2)
            self.assertEqual(r["small_masks_lt_64px"], 1)
            self.assertEqual(r["empty_bg_images"], 1)
            self.assertEqual(r["missing_labels"], 1)
