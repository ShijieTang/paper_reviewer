import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock


class ImmediateThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self.daemon = daemon

    def start(self):
        if self._target is not None:
            self._target()


class TestWebappCitationCheckDefault(unittest.TestCase):
    def _load_app_module(self):
        modular_seg_stub = types.ModuleType("modular_seg")
        modular_seg_stub.reconstruct_md = lambda sections: ""
        modular_seg_stub.save_sections = lambda *args, **kwargs: None
        modular_seg_stub.segment_md = lambda *args, **kwargs: {}
        sys.modules["modular_seg"] = modular_seg_stub

        doc_preprocess_stub = types.ModuleType("doc_preprocess")
        doc_preprocess_stub.doc_preprocess = MagicMock(return_value="data/md/test.md")
        doc_preprocess_stub.load_or_create_markdown = MagicMock(return_value="# Title\n")
        sys.modules["doc_preprocess"] = doc_preprocess_stub

        self.mas_loop_stub = types.ModuleType("mas_loop")
        self.mas_loop_stub.main = MagicMock(
            return_value={"reviewers": [], "conference": {}, "citations": {"stats": {"total": 1}}}
        )
        sys.modules["mas_loop"] = self.mas_loop_stub

        module_path = Path(__file__).resolve().parent.parent / "webapp" / "app.py"
        spec = importlib.util.spec_from_file_location("test_webapp_app", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        module.threading.Thread = ImmediateThread
        return module

    def test_api_run_keeps_citation_checker_enabled_by_default(self):
        app_module = self._load_app_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                md_dir = Path("data/md")
                md_dir.mkdir(parents=True, exist_ok=True)
                (md_dir / "test.md").write_text("# Title\n\nBody\n", encoding="utf-8")

                app_module.current_md_name = "test.md"
                client = app_module.app.test_client()

                response = client.post(
                    "/api/run",
                    data=json.dumps(
                        {
                            "topic": "NLP",
                            "reviewers": ["reviewer_a"],
                            "n_iter": 1,
                            "api_key": "key",
                        }
                    ),
                    content_type="application/json",
                )

                self.assertEqual(response.status_code, 200)
                self.mas_loop_stub.main.assert_called_once()
                self.assertNotIn("run_citation_check", self.mas_loop_stub.main.call_args.kwargs)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
