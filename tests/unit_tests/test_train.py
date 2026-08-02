import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _write_capture_command(path: Path) -> None:
    path.write_text(
        "#!/usr/bin/bash\n"
        'printf "%s\\n" "$@" > "${RWKV_CAPTURE_PATH}"\n'
        'printf "%s\\n" "${NGPU-}" "${LOCAL_RANK-}" '
        '"${TORCHFT_LIGHTHOUSE-}" > "${RWKV_CAPTURE_ENV_PATH}"\n',
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_run_train_uses_torchtitan_module_and_config_contract(tmp_path) -> None:
    capture_path = tmp_path / "torchrun-args"
    capture_env_path = tmp_path / "torchrun-env"
    fake_torchrun = tmp_path / "torchrun"
    _write_capture_command(fake_torchrun)
    environment = os.environ.copy()
    environment.update(
        {
            "CONFIG": "rwkv7_1_5b_infctx",
            "LOG_RANK": "0,1",
            "MODULE": "rwkv_lm.models.rwkv7",
            "NGPU": "4",
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "RWKV_CAPTURE_PATH": str(capture_path),
            "RWKV_CAPTURE_ENV_PATH": str(capture_env_path),
            "TORCHFT_LIGHTHOUSE": "http://lighthouse.test:29510",
        }
    )
    environment.pop("COMM_MODE", None)

    subprocess.run(
        [
            "/usr/bin/bash",
            str(ROOT / "run_train.sh"),
            "--training.steps",
            "2",
        ],
        check=True,
        env=environment,
    )

    assert capture_path.read_text(encoding="utf-8").splitlines() == [
        "--nproc_per_node=4",
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint=localhost:0",
        "--local-ranks-filter",
        "0,1",
        "--role",
        "rank",
        "--tee",
        "3",
        "-m",
        "torchtitan.train",
        "--module",
        "rwkv_lm.models.rwkv7",
        "--config",
        "rwkv7_1_5b_infctx",
        "--training.steps",
        "2",
    ]
    assert capture_env_path.read_text(encoding="utf-8").splitlines() == [
        "4",
        "",
        "http://lighthouse.test:29510",
    ]


def test_run_train_comm_mode_uses_torchtitan_debug_path(tmp_path) -> None:
    capture_path = tmp_path / "python-args"
    capture_env_path = tmp_path / "python-env"
    fake_python = tmp_path / "python3"
    _write_capture_command(fake_python)
    environment = os.environ.copy()
    environment.update(
        {
            "COMM_MODE": "local_tensor",
            "CONFIG": "rwkv7_debugmodel_infctx",
            "MODULE": "rwkv_lm.models.rwkv7",
            "NGPU": "16",
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "RWKV_CAPTURE_PATH": str(capture_path),
            "RWKV_CAPTURE_ENV_PATH": str(capture_env_path),
        }
    )

    subprocess.run(
        [
            "/usr/bin/bash",
            str(ROOT / "run_train.sh"),
            "--metrics.log-freq",
            "2",
        ],
        check=True,
        env=environment,
    )

    assert capture_path.read_text(encoding="utf-8").splitlines() == [
        "-m",
        "torchtitan.train",
        "--module",
        "rwkv_lm.models.rwkv7",
        "--config",
        "rwkv7_debugmodel_infctx",
        "--metrics.log-freq",
        "2",
        "--comm.mode=local_tensor",
        "--training.steps",
        "1",
    ]
    assert capture_env_path.read_text(encoding="utf-8").splitlines() == [
        "16",
        "0",
        "",
    ]
