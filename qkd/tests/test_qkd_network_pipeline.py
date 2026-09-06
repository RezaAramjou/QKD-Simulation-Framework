from pathlib import Path

import qkd_network_7nodes as network


def test_pipeline_loads_repository_local_engine() -> None:
    simulator = network.PipelineQKDLinkSimulator()

    assert simulator.available
    expected_path = Path(network.__file__).resolve().with_name("main_optimized.py")
    assert Path(simulator.pipeline_path).resolve() == expected_path


def test_pipeline_preserves_reference_efficiency() -> None:
    simulator = network.PipelineQKDLinkSimulator(detector_efficiency=0.15)
    config = simulator._make_pipeline_config()

    assert config["detector"]["det_eff_d0"] == 0.15
    assert config["detector"]["det_eff_d1"] == 0.15
    assert config["source"]["pulses"] == simulator._pipeline_modules[
        "DEFAULT_CFG"
    ]["source"]["pulses"]


def test_pipeline_forwards_requested_pulse_count() -> None:
    simulator = network.PipelineQKDLinkSimulator(
        detector_efficiency=0.15,
        apply_noise=True,
    )
    captured = {}

    def fake_init_worker(config, combo_map):
        captured["config"] = config
        captured["combo_map"] = combo_map

    def fake_run_single_simulation(args):
        captured["args"] = args
        return {
            "secure_key_bits": 123,
            "qber": 0.04,
            "secure_key_rate": 1.23e-5,
        }

    simulator._pipeline_modules["init_worker"] = fake_init_worker
    simulator._pipeline_modules["run_single_simulation"] = fake_run_single_simulation

    link = network.QKDLink("A-B", "A", "B", distance_km=30.0)
    result = simulator._run_pipeline(link, 10_000_000)

    assert result == (123, 0.04, 1.23e-5)
    assert captured["args"][0] == 30.0
    assert captured["args"][1] == 10_000_000
    assert captured["args"][2] is True
    assert captured["config"]["detector"]["det_eff_d0"] == 0.15
    assert captured["config"]["detector"]["det_eff_d1"] == 0.15


def test_local_pipeline_qber_is_below_bb84_threshold() -> None:
    simulator = network.PipelineQKDLinkSimulator(
        detector_efficiency=0.15,
        apply_noise=True,
    )
    config = simulator._make_pipeline_config()
    simulator._pipeline_modules["init_worker"](config, {0: {}})
    result = simulator._pipeline_modules["run_single_simulation"](
        (30.0, 100_000, True, 12345, 0)
    )

    assert result["total_pulses"] == 100_000
    assert result["qber"] < network.QBER_THRESHOLD
