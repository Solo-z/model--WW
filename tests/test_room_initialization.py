"""Initialization regression tests without GPUs, model weights, or third-party imports.

Run from the repository root:
    python -B tests/test_room_initialization.py

RoomEngine is imported normally: its model dependencies are lazy. The app cache
tests compile only the AST node for app.py's _get_engine function, avoiding the
module's Gradio imports, UI construction, and Space setup side effects.
"""

from __future__ import annotations

import ast
import logging
import sys
import traceback
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modelw import room  # noqa: E402


class RoomInitializationTests(unittest.TestCase):
    def setUp(self):
        self.state = types.SimpleNamespace(
            dit_mode="ok", lm_mode="ok", dit_calls=0, lm_calls=0
        )
        state = self.state

        class StubDit:
            def __init__(self):
                self.model = None
                self.vae = None
                self.text_tokenizer = None
                self.text_encoder = None

            def initialize_service(self, **kwargs):
                state.dit_calls += 1
                if state.dit_mode == "fail":
                    return "simulated DiT load failure", False
                if state.dit_mode != "missing_weights":
                    self.model = object()
                    self.vae = object()
                    self.text_tokenizer = object()
                    self.text_encoder = object()
                return "DiT loaded", True

        class StubLlm:
            def initialize(self, **kwargs):
                state.lm_calls += 1
                if state.lm_mode == "fail":
                    return "simulated LM load failure", False
                return "LM loaded", True

        package = types.ModuleType("acestep")
        package.__path__ = []
        handler = types.ModuleType("acestep.handler")
        handler.AceStepHandler = StubDit
        llm = types.ModuleType("acestep.llm_inference")
        llm.LLMHandler = StubLlm
        modules = patch.dict(
            sys.modules,
            {
                "acestep": package,
                "acestep.handler": handler,
                "acestep.llm_inference": llm,
            },
        )
        modules.start()
        self.addCleanup(modules.stop)
        checkpoint_dir = patch.object(
            room.RoomEngine, "_ace_step_checkpoint_dir", return_value="/fake/checkpoints"
        )
        checkpoint_dir.start()
        self.addCleanup(checkpoint_dir.stop)
        self.engine = room.RoomEngine(room.RoomConfig())

    def assert_not_cached(self):
        self.assertFalse(self.engine._initialized)
        self.assertIsNone(self.engine._acestep_dit)
        self.assertIsNone(self.engine._acestep_llm)

    def test_failed_dit_is_not_cached_and_retry_loads_both_handlers(self):
        self.state.dit_mode = "fail"
        with self.assertRaisesRegex(RuntimeError, "simulated DiT load failure"):
            self.engine.initialize()
        self.assert_not_cached()
        self.assertEqual(self.state.lm_calls, 0)

        self.state.dit_mode = "ok"
        self.engine.initialize()
        self.assertTrue(self.engine._initialized)
        self.assertEqual(self.state.dit_calls, 2)
        self.assertEqual(self.state.lm_calls, 1)

    def test_failed_lm_is_not_cached_and_retry_rebuilds_complete_pair(self):
        self.state.lm_mode = "fail"
        with self.assertRaisesRegex(RuntimeError, "simulated LM load failure"):
            self.engine.initialize()
        self.assert_not_cached()

        self.state.lm_mode = "ok"
        self.engine.initialize()
        self.assertTrue(self.engine._initialized)
        self.assertEqual(self.state.dit_calls, 2)
        self.assertEqual(self.state.lm_calls, 2)

    def test_reported_success_with_missing_weights_is_not_cached(self):
        self.state.dit_mode = "missing_weights"
        with self.assertRaises(RuntimeError):
            self.engine.initialize()
        self.assert_not_cached()

        self.state.dit_mode = "ok"
        self.engine.initialize()
        self.assertTrue(self.engine._initialized)
        self.assertEqual(self.state.dit_calls, 2)
        self.assertIsNotNone(self.engine._acestep_dit.model)
        self.assertIsNotNone(self.engine._acestep_llm)

    def test_successful_initialization_reuses_validated_handlers(self):
        self.engine.initialize()
        dit = self.engine._acestep_dit
        lm = self.engine._acestep_llm
        self.engine.initialize()

        self.assertTrue(self.engine._initialized)
        self.assertIs(self.engine._acestep_dit, dit)
        self.assertIs(self.engine._acestep_llm, lm)
        self.assertEqual(self.state.dit_calls, 1)
        self.assertEqual(self.state.lm_calls, 1)


class AppEngineCacheTests(unittest.TestCase):
    def setUp(self):
        self.instances = []
        self.should_fail = True
        owner = self

        class StubEngine:
            def __init__(self, config):
                self.ready = False
                self.initialize_calls = 0
                owner.instances.append(self)

            def initialize(self):
                self.initialize_calls += 1
                if owner.should_fail:
                    raise RuntimeError("simulated engine initialization failure")
                self.ready = True

        class GradioError(Exception):
            pass

        self.error_type = GradioError
        engine_class = patch.object(room, "RoomEngine", StubEngine)
        engine_class.start()
        self.addCleanup(engine_class.stop)

        # Extract exactly this function; never execute app.py module startup.
        tree = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_get_engine"
        )
        self.namespace = {
            "_engine": None,
            "_default_room_config": lambda: room.RoomConfig(),
            "gr": types.SimpleNamespace(Error=GradioError),
            "logging": logging,
            "traceback": traceback,
        }
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(ROOT / "app.py"), "exec"), self.namespace)
        self.get_engine = self.namespace["_get_engine"]

    def test_failed_initialization_is_not_published_and_retry_is_fresh(self):
        with self.assertRaises(self.error_type):
            self.get_engine()
        self.assertIsNone(self.namespace["_engine"])
        failed = self.instances[0]

        self.should_fail = False
        ready = self.get_engine()
        self.assertTrue(ready.ready)
        self.assertIsNot(ready, failed)
        self.assertIs(self.namespace["_engine"], ready)
        self.assertEqual(len(self.instances), 2)

    def test_successful_global_engine_is_reused(self):
        self.should_fail = False
        first = self.get_engine()
        second = self.get_engine()

        self.assertIs(first, second)
        self.assertTrue(first.ready)
        self.assertEqual(len(self.instances), 1)
        self.assertEqual(first.initialize_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
