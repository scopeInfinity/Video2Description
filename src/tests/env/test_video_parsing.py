import cv2
import unittest
import os

from tests.data import fetcher


class TestVideoParsing(unittest.TestCase):

    def test_opencv_videocapture(self):
        path = fetcher.get_videopath("12727.mp4")
        self.assertTrue(os.path.exists(path))
        vcap = cv2.VideoCapture(path)
        success_count = 0
        while True:
            success, _ = vcap.read()
            if not success:
                break
            success_count += 1
        self.assertGreater(success_count, 3*15)
        self.assertLess(success_count, 15*30)


if __name__ == '__main__':
    unittest.main()
