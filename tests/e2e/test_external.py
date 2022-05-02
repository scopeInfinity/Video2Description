import unittest
import json
import os

from parameterized import parameterized
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

WEB_URL = "http://localhost:8080"
ROOT_PATH = "src"
CONFIG_FILE = "src/config.json"


class TestExternal(unittest.TestCase):
    """Test from as a end user."""

    def setUp(self):
        options = Options()
        options.add_argument('-headless')
        self.driver = Firefox(options=options)

    def tearDown(self):
        self.driver.close()

    def get_video_path(self, fname):
        with open(CONFIG_FILE, "r") as fin:
            dir_videos = json.load(fin)["tests"]["dir_videos"]
            path = os.path.abspath(
                os.path.join(ROOT_PATH, dir_videos, fname))
            self.assertTrue(os.path.exists(path))
            return path

    @parameterized.expand([
        ("12727.mp4", "two men are talking about a cooking show"),
        ("12968.mp4", "a woman is talking about a makeup face"),
    ])
    def test_upload_and_verify(self, fname, caption):
        """
        Tests uploading a video and verify the response.
        Note: The tests values are currently hard coded to a specific trained
              model and it might fail for other models.
        """
        self.driver.get(WEB_URL)
        video_path = self.get_video_path(fname)
        text_vprocessing = "Video is being uploaded and processed"
        self.driver.find_element_by_xpath("//input[@type='file']").send_keys(video_path)
        self.assertNotIn(text_vprocessing,
                         self.driver.find_element_by_id("notifications").text)
        self.driver.find_element_by_xpath("//input[@value='Upload Video']").click()
        self.assertIn(text_vprocessing,
                      self.driver.find_element_by_id("notifications").text)
        WebDriverWait(self.driver, 120).until(
            EC.text_to_be_present_in_element(
                (By.ID, "notifications"), "Request Completed")
        )
        self.assertIn(caption, self.driver.find_element_by_id("results").text.lower())
