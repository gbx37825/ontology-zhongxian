import unittest

from src.simulation.engine.pipedream.wxq_to_superlink import parse_wxq_json


def _section(z_inv: float, location: float) -> dict:
    return {
        "location": location,
        "num_interp": 0,
        "yz": [[0.0, z_inv + 1.0], [1.0, z_inv], [2.0, z_inv]],
    }


class PipedreamFlowDirectionTests(unittest.TestCase):
    def test_channel_discretization_records_hydroos_flow_sign(self):
        model = {
            "baseData": {
                "nodes": {
                    "A": {"zb": 0.0, "Amin": 10.0},
                    "B": {"zb": 1.0, "Amin": 10.0},
                    "C": {"zb": 2.0, "Amin": 10.0},
                    "D": {"zb": 1.0, "Amin": 10.0},
                },
                "channels": {
                    "uphill": {"node1": "A", "node2": "B", "sec_names": ["A-1", "B-1"], "nc": 0.013},
                    "downhill": {"node1": "C", "node2": "D", "sec_names": ["C-1", "D-1"], "nc": 0.013},
                },
                "sections": {
                    "A-1": _section(0.0, 0.0),
                    "B-1": _section(1.0, 100.0),
                    "C-1": _section(2.0, 0.0),
                    "D-1": _section(1.0, 100.0),
                },
            },
            "initialData": {
                "channels": {"local_Z": {"uphill": 3.0, "downhill": 3.0}},
                "boundaries": {},
            },
        }

        converted = parse_wxq_json(model)

        self.assertEqual(converted["channel_discretization"]["uphill"]["flow_sign"], -1.0)
        self.assertEqual(converted["channel_discretization"]["uphill"]["sections"][0]["flow_sign"], -1.0)
        self.assertEqual(converted["channel_discretization"]["downhill"]["flow_sign"], 1.0)
        self.assertEqual(converted["channel_discretization"]["downhill"]["sections"][0]["flow_sign"], 1.0)


if __name__ == "__main__":
    unittest.main()
