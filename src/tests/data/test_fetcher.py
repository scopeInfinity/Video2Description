import unittest
import os

from tests.data import fetcher


class TestFetcher(unittest.TestCase):

    def test_get_videopath_success(self):
        path = fetcher.get_videopath(".content")
        self.assertTrue(os.path.exists(path))
        with open(path, "r") as f:
            self.assertEqual("I_AM_VIDEO_TESTDATA_DIR", f.read().strip())

    def test_get_videopath_failure(self):
        path = fetcher.get_videopath("bad_filename.mp4")
        self.assertFalse(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
