import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
TESTS_ROOT = PROJECT_ROOT / "tests"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))


def pytest_collection_modifyitems(session, config, items):  # type: ignore[no-untyped-def]
    json_test_suffix = "test_official_onnx_expected_errors"
    md_test_suffix = "test_official_onnx_file_support_doc"

    json_index = next(
        (index for index, item in enumerate(items) if item.nodeid.endswith(json_test_suffix)),
        None,
    )
    md_index = next(
        (index for index, item in enumerate(items) if item.nodeid.endswith(md_test_suffix)),
        None,
    )
    if json_index is None or md_index is None or json_index < md_index:
        return

    md_item = items.pop(md_index)
    if md_index < json_index:
        json_index -= 1
    items.insert(json_index + 1, md_item)
