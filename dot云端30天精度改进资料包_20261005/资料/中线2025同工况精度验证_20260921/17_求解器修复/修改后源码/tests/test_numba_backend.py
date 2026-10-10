from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import wxq_pipedream.hydraulic_engine as engine_module
from pipedream_solver.superlink import SuperLink
from pipedream_solver.nsuperlink import nSuperLink, gamma_uk, gamma_dk, numba_b_ik
from wxq_pipedream.backends import adapt_numba_model, solver_class


def variable_reaches():
    source = next(iter(json.loads((ROOT / "examples/示例模型.json").read_text()).values()))
    template = deepcopy(source["baseData"]["sections"])
    source["baseData"]["sections"] = {}
    source["baseData"]["channels"] = {}
    for index, links in enumerate((2, 5, 3)):
        up, down = f"upper-{index}", f"lower-{index}"
        upper = deepcopy(template["section-up"])
        lower = deepcopy(template["section-down"])
        upper["num_interp"] = links - 1
        lower["location"] = 100.0 + 25.0 * index
        source["baseData"]["sections"].update({up: upper, down: lower})
        source["baseData"]["channels"][f"reach-{index}"] = {
            "node1": "upstream", "node2": "downstream", "sec_names": [up, down], "nc": 0.03}
    source["initialData"]["boundaries"] = {"upstream": 1.0, "downstream": 3.0}
    return source


class NumbaEquivalenceTests(unittest.TestCase):
    def test_variable_lengths_require_exclusive_prefix(self):
        count = np.array([2, 5, 3])
        original = np.cumsum(count - 1) - (count[0] - 1)
        fixed = np.r_[0, np.cumsum(count[:-1] - 1)]
        np.testing.assert_array_equal(original, [0, 4, 6])
        np.testing.assert_array_equal(fixed, [0, 1, 5])
        self.assertTrue(np.any(original + count - 1 > np.sum(count - 1)))
        self.assertEqual(fixed[-1] + count[-1] - 1, np.sum(count - 1))

    def test_uniform_lengths_hide_original_offset_error(self):
        count = np.array([4, 4, 4])
        np.testing.assert_array_equal(np.cumsum(count - 1) - (count[0] - 1),
                                      np.r_[0, np.cumsum(count[:-1] - 1)])

    def test_local_loss_conventions_are_equal_only_after_conversion(self):
        q = np.array([-10.0, 0.0, 30.0])
        coefficient = np.array([0.5, 0.9, 1 / 0.67**2])
        area = np.array([5.0, 8.0, 12.0])
        loss = 1 / coefficient**2
        standard_up = SuperLink.gamma_uk(None, q, coefficient, area)
        standard_down = SuperLink.gamma_dk(None, q, coefficient, area)
        np.testing.assert_allclose(standard_up, gamma_uk(q, loss, area), atol=1e-14)
        np.testing.assert_allclose(standard_down, gamma_dk(q, loss, area), atol=1e-14)
        self.assertFalse(np.allclose(standard_up, gamma_uk(q, coefficient, area)))

    def test_internal_control_loss_convention(self):
        dx = np.array([100.0, 200.0])
        roughness = np.array([0.03, 0.025])
        flow = np.array([10.0, -5.0])
        area = np.array([8.0, 9.0])
        radius = np.array([2.0, 2.5])
        control_area = np.array([2.0, 4.0])
        coefficient = np.array([0.65, 0.85])
        a, c = np.zeros(2), np.zeros(2)
        control = np.ones(2, dtype=bool)
        expected = SuperLink.b_ik(None, dx, 15.0, roughness, flow, area, radius,
                                 control_area, coefficient, a, c, control)
        actual = numba_b_ik(dx, 15.0, roughness, flow, area, radius, control_area,
                           1 / coefficient**2, a, c, control, np.ones(2), np.zeros(2, dtype=np.int64), 9.81)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_corrected_variable_reaches_match_standard(self):
        engines = []
        for backend in ("standard", "numba"):
            engine = engine_module.PipedreamHydraulicEngine.from_wxq(variable_reaches(), dt=15.0, backend=backend)
            model = engine.model
            model._method = "lsq"
            np.testing.assert_array_equal(model.nk, [2, 5, 3])
            if backend == "numba":
                np.testing.assert_array_equal(model._k_1k, [0, 1, 5])
            engines.append(engine)
        for _ in range(100):
            for engine in engines:
                engine.step(boundary_heads={"downstream": 3}, node_inflows={"upstream": 1})
            np.testing.assert_allclose(engines[0].model.H_j, engines[1].model.H_j, atol=1e-7)
            np.testing.assert_allclose(engines[0].model.h_Ik, engines[1].model.h_Ik, atol=1e-7)
            np.testing.assert_allclose(engines[0].model.Q_ik, engines[1].model.Q_ik, atol=1e-6)

    def engine(self, backend="numba"):
        return engine_module.PipedreamHydraulicEngine.from_wxq(variable_reaches(), dt=15, backend=backend)

    def test_disabled_jit_is_not_silently_reported_as_acceleration(self):
        with patch("numba.config.DISABLE_JIT", 1):
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                solver_class("numba")
            self.assertIs(solver_class("standard"), SuperLink)

    def test_standard_does_not_require_numba_import(self):
        with patch.dict(sys.modules, {"numba": None}):
            self.assertIs(solver_class("standard"), SuperLink)
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                solver_class("numba")

    def test_unknown_backend_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine("numbba")

    def test_local_loss_mapping_does_not_change_gate_discharge_coefficient(self):
        source = variable_reaches()
        source["baseData"]["gates"]["test-gate"] = {
            "node1": "upstream", "node2": "downstream", "b": 2, "c1": 0.65, "zb": 0,
        }
        ordinary = engine_module.PipedreamHydraulicEngine.from_wxq(source, backend="standard")
        fast = engine_module.PipedreamHydraulicEngine.from_wxq(source)
        np.testing.assert_array_equal(fast.model._Co, ordinary.model._Co)
        np.testing.assert_array_equal(fast.model._Co, [0.65])
        for name in ("_C_uk", "_C_dk", "_C_ik"):
            np.testing.assert_array_equal(getattr(fast.model, name), 1 / getattr(ordinary.model, name)**2)

    def test_invalid_loss_parameter_is_rejected_before_adaptation(self):
        for bad in (0.0, -1.0, np.nan, np.inf, 1e-300, 1e300):
            with self.subTest(value=bad):
                model = self.engine("standard").model
                before = model._k_1k.copy()
                model._C_uk[0] = bad
                with self.assertRaises(ValueError):
                    adapt_numba_model(model)
                np.testing.assert_array_equal(model._k_1k, before)

    def test_unsupported_layout_or_friction_is_rejected(self):
        for attribute, value in (("_bc_method", "b"), ("_Sf_method_ik", np.ones(10, dtype=int))):
            model = self.engine("standard").model
            setattr(model, attribute, value)
            with self.assertRaises(ValueError):
                adapt_numba_model(model)
        model = self.engine("standard").model
        model._ki[:] = model._ki[::-1]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            adapt_numba_model(model)

    def test_same_backend_cache_is_idempotent_and_replays_lsq(self):
        for backend in ("standard", "numba"):
            with self.subTest(backend=backend):
                original = self.engine(backend)
                original.model._method = "lsq"
                for _ in range(20):
                    original.step(boundary_heads={"downstream": 3}, node_inflows={"upstream": 1})
                saved = original.get_restart_state()
                restored = self.engine(backend)
                restored.load_restart_state(saved)
                restored.load_restart_state(saved)
                self.assertEqual(restored.model._method, "lsq")
                for name in ("_k_1k", "_C_uk", "_C_dk", "_C_ik"):
                    np.testing.assert_array_equal(getattr(original.model, name), getattr(restored.model, name))
                for index in range(10):
                    for engine in (original, restored):
                        engine.step(boundary_heads={"downstream": 3 + index * 0.01}, node_inflows={"upstream": 1.2})
                    np.testing.assert_allclose(original.model.H_j, restored.model.H_j, rtol=0, atol=1e-10)
                    np.testing.assert_allclose(original.model.Q_ik, restored.model.Q_ik, rtol=0, atol=1e-9)

    def test_cross_backend_legacy_and_changed_dt_caches_are_rejected(self):
        for backend, other in (("numba", "standard"), ("standard", "numba")):
            engine = self.engine(backend)
            saved = self.engine(other).get_restart_state()
            before = engine.model.H_j.copy()
            with self.assertRaisesRegex(ValueError, "拒绝混用"):
                engine.load_restart_state(saved)
            np.testing.assert_array_equal(engine.model.H_j, before)
        engine = self.engine()
        saved = engine.get_restart_state()
        saved["reference_dt_s"] = 30
        with self.assertRaises(ValueError):
            engine.load_restart_state(saved)
        legacy = self.engine("standard").get_restart_state()
        legacy["schema_version"] = "pipedream-superlink-restart-state-v1"
        legacy.pop("solver")
        with self.assertRaisesRegex(ValueError, "旧预热缓存"):
            engine.load_restart_state(legacy, allow_legacy=True)
        standard = self.engine("standard")
        with self.assertRaises(ValueError):
            standard.load_restart_state(legacy)
        standard.load_restart_state(legacy, allow_legacy=True)

    def test_forged_cache_does_not_overwrite_adapted_parameters(self):
        engine = self.engine()
        for name in ("_C_uk", "_C_dk", "_C_ik", "_k_1k"):
            saved = deepcopy(engine.get_restart_state())
            saved["arrays"][name]["values"][0] += 1
            before = engine.model.H_j.copy()
            with self.assertRaisesRegex(ValueError, "拒绝混用"):
                engine.load_restart_state(saved)
            np.testing.assert_array_equal(engine.model.H_j, before)
        for change in ("missing_head", "invalid_scalar", "size", "method", "depth_limit"):
            saved = deepcopy(engine.get_restart_state())
            if change == "missing_head":
                saved["arrays"].pop("H_j")
            elif change == "size":
                saved["scalars"]["NK"] += 1
            elif change == "method":
                saved.pop("internal_method")
            elif change == "depth_limit":
                saved["arrays"]["max_depth"]["values"][0] = np.nan
            else:
                saved["scalars"]["step"] = 0
            with self.assertRaises(ValueError):
                engine.load_restart_state(saved)

    def test_cli_declares_default_and_blocks_cross_backend_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = [sys.executable, str(ROOT / "scripts/run_simulation.py"), str(ROOT / "examples/示例模型.json")]
            flags = ["--warmup-max-steps", "1", "--max-compute-steps", "1"]
            first = subprocess.run(base + [str(root / "default")] + flags, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            summary = json.loads((root / "default/运行摘要.json").read_text())
            self.assertEqual(summary["solver"]["backend"], "numba")
            self.assertEqual(summary["solver"]["class"], "pipedream_solver.nsuperlink.nSuperLink")
            self.assertIn("SOLVER", first.stdout)
            second = subprocess.run(base + [str(root / "cross")] + flags + ["--backend", "standard",
                "--restart-state", str(root / "default/预热末态.json")], capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("拒绝混用", second.stderr)
            self.assertFalse((root / "cross/Pipedream原始输出.csv").exists())



class RectangularGateRegressionTests(unittest.TestCase):
    def test_pressure_integral_limits_and_continuity(self):
        from pipedream_solver.gate_hydraulics import rectangular_gate_discharge as q
        for fn in [q, q.py_func]:
            self.assertEqual(fn(6., 6., 8., 6.4, .7), 0.)
            self.assertEqual(fn(6., 5., 0., 6.4, .7), 0.)
            self.assertEqual(fn(-1., -2., 2., 6.4, .7), 0.)
            self.assertEqual(fn(5., 6., 2., 6.4, .7, True), 0.)
            self.assertAlmostEqual(fn(6., 5., 2., 6.4, .7), .7 * 6.4 * 2 * np.sqrt(2 * 9.81))
            self.assertAlmostEqual(fn(6., 5., 2., 6.4, .7), -fn(5., 6., 2., 6.4, .7))
            self.assertLess(abs(fn(6., 5.5, 6.-1e-7, 6.4, .7)-fn(6., 5.5, 6.+1e-7, 6.4, .7)), 1e-5)
            for hu, hd, opening in [(6., 5.5, 5.75), (6., 0., 3.), (6., 0., 8.)]:
                top = min(hu, opening)
                z = (np.arange(100000) + .5) * top / 100000
                expected = .7 * 6.4 * np.sqrt(2*9.81) * np.sqrt(hu - np.maximum(hd, z)).mean() * top
                self.assertAlmostEqual(fn(hu, hd, opening, 6.4, .7), expected, places=5)

    def test_converter_geometry_and_both_backends(self):
        from wxq_pipedream.wxq_to_superlink import parse_wxq_json
        source = next(iter(json.loads((ROOT / 'examples/示例模型.json').read_text()).values()))
        source['baseData']['gates'] = {'gate': {'node1': 'upstream', 'node2': 'downstream', 'zb': 0., 'b': 6.4, 'c1': .7}}
        converted = parse_wxq_json(source, gate_max_openings={'gate': 7.787}, section_geometry_mode='safe')
        gate = converted['orifices'].iloc[0]
        self.assertEqual(gate.g1, 7.787)
        self.assertEqual(gate.g2, 6.4)
        engines = [engine_module.PipedreamHydraulicEngine(deepcopy(converted), dt=15., backend=b) for b in ['standard', 'numba']]
        models = [engine.model for engine in engines]
        values = []
        for m in models:
            m.H_j[:] = 6.
            m.H_j[m._J_do] = 5.5
            m.compute_storage_volumes()
            m.orifice_hydraulic_geometry(u=np.ones(1))
            m.orifice_flow_coefficients(u=np.ones(1))
            m.solve_orifice_flows(dt=.001, u=np.ones(1))
            qlinear = m._alpha_o * m.H_j[m._J_uo] + m._beta_o * m.H_j[m._J_do] + m._chi_o
            np.testing.assert_allclose(m.Q_o, qlinear, atol=1e-9)
            values.append(m.Q_o.copy())
            old_t, old_i, old_h = m.t, m.iter_count, m.H_j.copy()
            with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                m.step(H_bc=m.H_j.copy(), Q_in=np.zeros(m.M), u_o=np.ones(1), dt=15., num_iter=1, head_tol=0., strict=True)
            self.assertEqual(m.t, old_t)
            self.assertEqual(m.iter_count, old_i)
            np.testing.assert_array_equal(m.H_j, old_h)
        np.testing.assert_allclose(values[0], values[1], atol=1e-10)
        for engine in engines:
            old_t = engine.model.t
            with self.assertRaisesRegex(RuntimeError, 'did not converge'):
                engine.step(gate_openings={'gate': 1.0}, head_tol=0.0)
            self.assertEqual(engine.model.iter_elapsed, 20)
            self.assertEqual(engine.model.t, old_t)

if __name__ == "__main__":
    unittest.main()
