import unittest
import os

import config

class TestConfig(unittest.TestCase):

    def setUp(self):
        config.clear()
        if 'V2D_CONFIG_FILE' in os.environ:
            del os.environ['V2D_CONFIG_FILE']

    def test_json(self):
        self.assertTrue(config.getConfig())

    def test_json_docker(self):
        os.environ['V2D_CONFIG_FILE'] = 'config_docker.json'
        self.assertTrue(config.getConfig())

    def test_json_bad_file(self):
        os.environ['V2D_CONFIG_FILE'] = 'config_bad_filename.json'
        with self.assertRaises(IOError):
            config.getConfig()

if __name__ == '__main__':
    unittest.main()
