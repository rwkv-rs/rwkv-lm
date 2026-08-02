from types import SimpleNamespace

from rwkv_lm.models.rwkv7 import parallelize as parallelize_module


def test_parallelize_order_matches_torchtitan(monkeypatch) -> None:
    events = []
    model = object()

    class ActivationCheckpointing:
        def apply(self, actual_model) -> None:
            assert actual_model is model
            events.append("activation_checkpoint")

    class ActivationCheckpointingConfig:
        def build(self, *, dump_folder):
            assert dump_folder == "outputs"
            return ActivationCheckpointing()

    monkeypatch.setattr(
        parallelize_module,
        "apply_compile",
        lambda actual_model, _: events.append("compile"),
    )

    def apply_fsdp(actual_model, mesh, **kwargs) -> None:
        assert actual_model is model
        assert mesh == "fsdp-mesh"
        assert kwargs["pp_enabled"] is False
        events.append("fsdp")

    monkeypatch.setattr(
        parallelize_module,
        "apply_fsdp_to_decoder",
        apply_fsdp,
    )
    parallel_dims = SimpleNamespace(
        tp_enabled=False,
        cp_enabled=False,
        dp_replicate_enabled=False,
        pp_enabled=False,
        get_mesh=lambda names: "fsdp-mesh",
    )
    training = SimpleNamespace(
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        enable_cpu_offload=False,
    )
    parallelism = SimpleNamespace(
        spmd_backend="default",
        fsdp_reshard_after_forward="default",
        enable_fsdp_symm_mem=False,
    )
    compile_config = SimpleNamespace(enable=True, components=["model"])

    result = parallelize_module.parallelize_rwkv7(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=ActivationCheckpointingConfig(),
        dump_folder="outputs",
    )

    assert result is model
    assert events == ["activation_checkpoint", "compile", "fsdp"]


def test_full_dtensor_validates_and_applies_declarative_sharding(monkeypatch) -> None:
    events = []

    class Model:
        def parallelize(self, parallel_dims) -> None:
            assert parallel_dims is dims
            events.append("model_parallelize")

    model = Model()
    dims = SimpleNamespace(
        tp_enabled=False,
        cp_enabled=False,
        dp_replicate_enabled=False,
        pp_enabled=False,
    )
    monkeypatch.setattr(
        parallelize_module,
        "validate_config",
        lambda actual_dims, actual_model: events.append(
            "validate"
            if actual_dims is dims and actual_model is model
            else "invalid_validate"
        ),
    )
    monkeypatch.setattr(
        parallelize_module,
        "resolve_fsdp_mesh",
        lambda actual_dims: (
            ("fsdp-mesh", ("fsdp",)) if actual_dims is dims else ("invalid", None)
        ),
    )
    monkeypatch.setattr(
        parallelize_module,
        "apply_compile",
        lambda actual_model, _: events.append("compile"),
    )
    monkeypatch.setattr(
        parallelize_module,
        "apply_fsdp_to_decoder",
        lambda actual_model, mesh, **kwargs: (
            events.append("fsdp")
            if actual_model is model
            and mesh == "fsdp-mesh"
            and kwargs["dp_mesh_dims"] == ("fsdp",)
            else events.append("invalid_fsdp")
        ),
    )
    training = SimpleNamespace(
        mixed_precision_param="bfloat16",
        mixed_precision_reduce="float32",
        enable_cpu_offload=False,
    )
    parallelism = SimpleNamespace(
        spmd_backend="full_dtensor",
        fsdp_reshard_after_forward="default",
        enable_fsdp_symm_mem=False,
    )
    compile_config = SimpleNamespace(enable=True, components=["model"])

    parallelize_module.parallelize_rwkv7(
        model,
        parallel_dims=dims,
        training=training,
        parallelism=parallelism,
        compile_config=compile_config,
        ac_config=None,
        dump_folder="outputs",
    )

    assert events == ["validate", "model_parallelize", "compile", "fsdp"]
