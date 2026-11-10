import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def pytest_configure(config) -> None:
    # The official-file tests are parametrized from the files on disk during
    # collection, so the node tests must exist before any xdist worker starts.
    if hasattr(config, "workerinput"):
        return
    from onnx_node_test_data import NodeTestDataError, ensure_node_test_data

    try:
        ensure_node_test_data()
    except NodeTestDataError as error:
        config.issue_config_time_warning(
            UserWarning(f"{error} ONNX node backend tests are unavailable."),
            stacklevel=2,
        )
