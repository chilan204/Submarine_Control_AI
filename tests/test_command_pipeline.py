import unittest
import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

# The bundled verification interpreter intentionally has no project packages.
# Supply a minimal requests module because every network call is mocked here.
try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    requests_stub = ModuleType("requests")
    requests_stub.get = MagicMock()
    sys.modules["requests"] = requests_stub

from core import command_cache, parser


class CommandPipelineTests(unittest.TestCase):
    def setUp(self):
        with command_cache.cache_lock:
            command_cache.COMMAND_CACHE.clear()
            command_cache.COMMAND_CACHE.extend([
                {
                    "keyword": "tiến lên",
                    "action": "MOVE",
                    "direction": "FORWARD",
                    "hasValue": False,
                },
                {
                    "keyword": "lùi lại",
                    "action": "MOVE",
                    "direction": "BACKWARD",
                    "hasValue": False,
                },
            ])

    def tearDown(self):
        with command_cache.cache_lock:
            command_cache.COMMAND_CACHE.clear()

    def test_parser_uses_normalized_exact_match(self):
        self.assertEqual(
            parser.parse_command("  TIẾN   LÊN "),
            {"action": "MOVE", "direction": "FORWARD"},
        )
        self.assertIsNone(parser.parse_command("hãy tiến lên"))
        self.assertIsNone(parser.parse_command(""))

    def test_failed_reload_preserves_last_known_good_cache(self):
        with patch.object(
            command_cache,
            "load_command_db",
            side_effect=RuntimeError("upstream unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                command_cache.reload_command_cache()

        self.assertEqual(len(command_cache.get_command_cache()), 2)

    @patch.dict("os.environ", {"COMMAND_API_URL": "http://commands"}, clear=False)
    @patch("core.command_cache.requests.get")
    def test_blank_keyword_is_rejected(self, mock_get):
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {
            "data": [{"keyword": "", "action": "MOVE"}]
        }

        with self.assertRaises(ValueError):
            command_cache.load_command_db()


if __name__ == "__main__":
    unittest.main()
