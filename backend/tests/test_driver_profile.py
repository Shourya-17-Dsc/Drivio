from app.analysis.fatigue import FatigueAnalyzer
from app.config import Settings
from app.pipeline import MonitoringPipeline
from helpers import feats


def test_pipeline_loads_driver_calibration_without_recalibrating():
    analyzer = FatigueAnalyzer(Settings())
    for i in range(100):
        analyzer.update(i / 30, feats())
    calibration = analyzer.calibration_state()
    pipe = MonitoringPipeline(Settings(enable_audio=False))
    pipe.set_driver(7, "Test driver", calibration)

    assert pipe.driver_id == 7
    assert pipe.driver_name == "Test driver"
    assert pipe.analyzer.calibrated
    assert pipe.analyzer.calibration_state() == calibration
