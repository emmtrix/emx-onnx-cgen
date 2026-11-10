import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"
BACKEND_TEST_DATA_GENERATOR = PROJECT_ROOT / "onnx-backend-test-data" / "generate.py"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def _load_backend_test_data_generator():
    # generate.py lives next to the data it builds, outside of any package.
    spec = importlib.util.spec_from_file_location(
        "onnx_backend_test_data_generate", BACKEND_TEST_DATA_GENERATOR
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_configure(config) -> None:
    # The official-file tests are parametrized from the files on disk during
    # collection, so the backend test data must exist before any xdist worker
    # starts.
    if hasattr(config, "workerinput"):
        return
    generator = _load_backend_test_data_generator()
    try:
        generator.ensure_backend_test_data()
    except generator.BackendTestDataError as error:
        config.issue_config_time_warning(
            UserWarning(f"{error} Official ONNX backend tests are unavailable."),
            stacklevel=2,
        )
