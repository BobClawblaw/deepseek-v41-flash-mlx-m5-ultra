import unittest
from pathlib import Path

from mac import config, launchd


class ConfigTest(unittest.TestCase):
    def test_env_overrides_and_serve_args(self):
        s = config.load(Path("/nonexistent"), env={"PORT": "9100", "PARALLEL": "1", "VISION": "false", "TENSORFOLD_SHA": "abc"})
        self.assertEqual(s.port, 9100)
        a = s.serve_args()
        self.assertIn("--parallel", a)
        self.assertEqual(a[a.index("--parallel") + 1], "1")
        self.assertNotIn("--vision", a)
        self.assertEqual(a[-1], str(s.model))
        self.assertIn("--prompt-cache-over-cap", a)

    def test_vision_urls_is_opt_in(self):
        self.assertNotIn("--vision-urls", config.load(Path("/nonexistent"), env={}).serve_args())
        self.assertIn("--vision-urls", config.load(Path("/nonexistent"), env={"VISION_URLS": "true"}).serve_args())
        self.assertNotIn("--vision-urls", config.load(Path("/nonexistent"), env={"VISION_URLS": "true", "VISION": "false"}).serve_args())

    def test_thinking_reserve_is_32k(self):
        a = config.load(Path("/nonexistent"), env={}).serve_args()
        self.assertEqual(a[a.index("--thinking-reserve") + 1], "32768")
        self.assertNotIn("--thinking-reserve", config.load(Path("/nonexistent"), env={"THINKING_RESERVE": "0"}).serve_args())

    def test_validate_bounds_parallel(self):
        s = config.load(Path("/nonexistent"), env={"PARALLEL": "9", "BINARY": "/x"})
        self.assertTrue(any("parallel" in p for p in config.validate(s)))

    def test_launchd_unit_serves_the_lan_on_the_port(self):
        s = config.load(Path("/nonexistent"), env={})
        text = launchd.plist(s, 8888)
        self.assertIn("<key>PORT</key><string>8888</string>", text)
        self.assertIn("<string>--foreground</string>", text)


if __name__ == "__main__":
    unittest.main()
