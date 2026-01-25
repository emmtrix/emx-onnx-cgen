import os

from emx_onnx_cgen import determinism


def test_deterministic_reference_runtime_restores_env() -> None:
    env_vars = determinism.THREAD_ENV_VARS
    if not env_vars:
        raise AssertionError("Expected thread env vars to be defined")
    first_var = env_vars[0]
    second_var = env_vars[1]
    os.environ[first_var] = "7"
    os.environ.pop(second_var, None)

    with determinism.deterministic_reference_runtime():
        assert os.environ[first_var] == "1"
        assert os.environ[second_var] == "1"

    assert os.environ[first_var] == "7"
    assert second_var not in os.environ
