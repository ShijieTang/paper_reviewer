import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from agents import DEFAULT_MODELS


class ImmediateThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self.daemon = daemon

    def start(self):
        if self._target is not None:
            self._target()


class TestWebappCitationCheckDefault(unittest.TestCase):
    def setUp(self):
        self._module_backups = {}
        self._thread_class_backup = None

    def tearDown(self):
        if self._thread_class_backup is not None:
            import threading

            threading.Thread = self._thread_class_backup
        for name, original in reversed(list(self._module_backups.items())):
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    def _replace_module(self, name, module):
        if name not in self._module_backups:
            self._module_backups[name] = sys.modules.get(name)
        sys.modules[name] = module

    def _load_app_module(self):
        modular_seg_stub = types.ModuleType("modular_seg")
        modular_seg_stub.reconstruct_md = lambda sections: ""
        modular_seg_stub.save_sections = lambda *args, **kwargs: None
        modular_seg_stub.segment_md = lambda *args, **kwargs: {}
        self._replace_module("modular_seg", modular_seg_stub)

        doc_preprocess_stub = types.ModuleType("doc_preprocess")
        doc_preprocess_stub.doc_preprocess = MagicMock(return_value="data/md/test.md")
        doc_preprocess_stub.load_or_create_markdown = MagicMock(return_value="# Title\n")
        self._replace_module("doc_preprocess", doc_preprocess_stub)

        self.mas_loop_stub = types.ModuleType("mas_loop")
        self.mas_loop_stub.main = MagicMock(
            return_value={"reviewers": [], "conference": {}, "citations": {"stats": {"total": 1}}}
        )
        self._replace_module("mas_loop", self.mas_loop_stub)

        module_path = Path(__file__).resolve().parent.parent / "webapp" / "app.py"
        spec = importlib.util.spec_from_file_location("test_webapp_app", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        if self._thread_class_backup is None:
            self._thread_class_backup = module.threading.Thread
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
                            "api_key": "sk-test",
                        }
                    ),
                    content_type="application/json",
                )

                self.assertEqual(response.status_code, 200)
                self.mas_loop_stub.main.assert_called_once()
                self.assertNotIn("run_citation_check", self.mas_loop_stub.main.call_args.kwargs)
            finally:
                os.chdir(old_cwd)

    def test_gemini_default_model_is_current_flash_generation(self):
        self.assertEqual(DEFAULT_MODELS["gemini"], "gemini-3.5-flash")

    def test_provider_endpoint_lists_deepseek_and_qwen(self):
        app_module = self._load_app_module()
        response = app_module.app.test_client().get("/api/providers")
        self.assertEqual(response.status_code, 200)
        providers = {item["id"]: item for item in response.get_json()["providers"]}
        self.assertEqual(providers["deepseek"]["default_model"], "deepseek-v4-flash")
        self.assertEqual(providers["qwen"]["default_model"], "qwen3.7-plus")

    def test_run_rag_rejects_non_cmu_gateway_key_before_llm_fallback(self):
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
                    "/api/run-rag",
                    data=json.dumps(
                        {
                            "topic": "NLP",
                            "api_key": "AQ.Abad-key",
                            "provider": "cmu",
                            "model": "",
                        }
                    ),
                    content_type="application/json",
                )

                self.assertEqual(response.status_code, 400)
                self.assertIn("starts with 'sk-'", response.get_json()["error"])
            finally:
                os.chdir(old_cwd)

    def test_run_rag_then_review_reuses_matching_package(self):
        app_module = self._load_app_module()

        rag_stub = types.ModuleType("rag")
        rag_package = {
            "rag_package_id": "rag_test",
            "warnings": [],
            "cutoff_report": {"cutoff_date": "2024-12-31"},
            "related_work_summary": "summary",
            "paper_metadata": [],
            "reranking_results": [],
        }
        rag_stub.build_rag_package = MagicMock(return_value=rag_package)
        old_rag_module = sys.modules.get("rag")
        sys.modules["rag"] = rag_stub

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = Path.cwd()
            os.chdir(tmpdir)
            try:
                md_dir = Path("data/md")
                md_dir.mkdir(parents=True, exist_ok=True)
                (md_dir / "test.md").write_text("# Title\n\nBody\n", encoding="utf-8")

                app_module.current_md_name = "test.md"
                client = app_module.app.test_client()

                rag_response = client.post(
                    "/api/run-rag",
                    data=json.dumps(
                        {
                            "api_key": "sk-test",
                            "provider": "cmu",
                            "model": "",
                        }
                    ),
                    content_type="application/json",
                )
                self.assertEqual(rag_response.status_code, 200)
                self.assertEqual(rag_response.get_json()["rag_package"]["rag_package_id"], "rag_test")
                self.assertEqual(rag_stub.build_rag_package.call_args.kwargs["topic"], "")

                run_response = client.post(
                    "/api/run",
                    data=json.dumps(
                        {
                            "topic": "NLP",
                            "reviewers": ["reviewer_a"],
                            "n_iter": 1,
                            "api_key": "sk-test",
                            "provider": "cmu",
                            "model": "",
                            "enable_rag": True,
                        }
                    ),
                    content_type="application/json",
                )

                self.assertEqual(run_response.status_code, 200)
                self.mas_loop_stub.main.assert_called_once()
                self.assertIs(self.mas_loop_stub.main.call_args.kwargs["precomputed_rag_package"], rag_package)
                self.assertTrue(self.mas_loop_stub.main.call_args.kwargs["enable_rag"])
            finally:
                os.chdir(old_cwd)
                if old_rag_module is None:
                    sys.modules.pop("rag", None)
                else:
                    sys.modules["rag"] = old_rag_module

if __name__ == "__main__":
    unittest.main()
