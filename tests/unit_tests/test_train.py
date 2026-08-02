import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_run_train_uses_torchtitan_module_and_config_contract(tmp_path) -> None:
    capture_path = tmp_path / "torchrun-args"
    fake_torchrun = tmp_path / "torchrun"
    fake_torchrun.write_text(
        '#!/usr/bin/bash\nprintf "%s\\n" "$@" > "${RWKV_CAPTURE_PATH}"\n',
        encoding="utf-8",
    )
    fake_torchrun.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "CONFIG": "rwkv7_1_5b_infctx",
            "LOG_RANK": "0,1",
            "MODULE": "rwkv_lm.models.rwkv7",
            "NGPU": "4",
            "PATH": f"{tmp_path}:{environment['PATH']}",
            "RWKV_CAPTURE_PATH": str(capture_path),
        }
    )

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
