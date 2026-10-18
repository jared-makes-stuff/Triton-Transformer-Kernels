"""Graph inference must preserve fresh values, state changes, and Tensor ownership."""
import copy

import pytest
import torch

from transformer_opt.kernels import triton_available
from transformer_opt.submission import TransformerConfig, UserOptimizedTransformer


def make_model(**kwargs):
    return UserOptimizedTransformer(TransformerConfig(1, 128, 128, 4, 128, 4, True), **kwargs).eval()


def test_graph_options_and_cpu_state_contract():
    with pytest.raises(ValueError, match="execution_mode"):
        make_model(execution_mode="invalid")
    model = make_model()
    cloned = copy.deepcopy(model)
    assert cloned.state_dict().keys() == model.state_dict().keys()
    with torch.inference_mode():
        assert not model._can_replay_cuda_graph(torch.randn(1, 128, 128), None)
    assert cloned._inference_graph is None


gpu = pytest.mark.skipif(not triton_available(), reason="CUDA Triton is unavailable")


@gpu
def test_replay_uses_new_inputs_and_masks_and_does_not_alias_outputs():
    model = make_model().cuda()
    eager = make_model(execution_mode="eager").cuda()
    eager.load_state_dict(model.state_dict(), strict=True)
    with torch.inference_mode():
        x = torch.randn(1, 128, 128, device="cuda")
        mask = torch.ones(1, 128, device="cuda", dtype=torch.bool)
        first = model(x, mask)
        saved = first.clone()
        entry = model._inference_graph
        for padding in (False, True):
            x.normal_()
            if padding:
                mask[:, 80:] = False
            actual = model(x, mask)
            assert torch.equal(actual, eager(x, mask))
            assert model._inference_graph is entry
            assert torch.equal(first, saved)
            assert actual.data_ptr() != first.data_ptr()
        assert model.cuda_graph_replays == 3
        assert model.attention_backend_counts == {"triton": 12, "sdpa": 0, "reference": 0}
        no_mask = model(x)
        assert torch.equal(no_mask, eager(x))
        assert model._inference_graph is not entry


@gpu
def test_replay_recaptures_after_parameter_update_state_load_and_epsilon_change():
    model = make_model().cuda()
    eager = make_model(execution_mode="eager").cuda()
    eager.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 128, 128, device="cuda")
    with torch.inference_mode():
        model(x)
        entry = model._inference_graph
        model.final_norm.bias.add_(0.125)
        eager.final_norm.bias.add_(0.125)
        assert torch.equal(model(x), eager(x))
        assert model._inference_graph is not entry
        entry = model._inference_graph
        model.layers[0].attention.q_proj.weight.add_(0.01)
        eager.layers[0].attention.q_proj.weight.add_(0.01)
        assert torch.equal(model(x), eager(x))
        assert model._inference_graph is not entry
        entry = model._inference_graph
        model.load_state_dict(eager.state_dict(), strict=True)
        assert torch.equal(model(x), eager(x))
        assert model._inference_graph is not entry
        entry = model._inference_graph
        model.final_norm.eps = eager.final_norm.eps = 0.02
        assert torch.equal(model(x), eager(x))
        assert model._inference_graph is not entry
        # Packed-weight signatures require ordinary versioned parameters.
        with torch.inference_mode(False):
            cloned = copy.deepcopy(model)
        assert cloned._inference_graph is None
        assert torch.equal(cloned(x), eager(x))
        model.clear_cuda_graph()
        assert model._inference_graph is None
        model(x)
        model.half()
        assert model._inference_graph is None
        assert model._packed_qkv_cache == {}


@gpu
def test_replay_keeps_separate_caller_streams_ordered():
    model = make_model().cuda()
    eager = make_model(execution_mode="eager").cuda()
    eager.load_state_dict(model.state_dict(), strict=True)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    with torch.inference_mode():
        inputs = [torch.randn(1, 128, 128, device="cuda") for _ in streams]
        model(inputs[0])
        outputs = []
        for stream, x in zip(streams, inputs):
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                outputs.append(model(x))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for x, actual in zip(inputs, outputs):
            assert torch.equal(actual, eager(x))


@gpu
def test_replay_recaptures_when_inference_context_changes():
    model = make_model().cuda()
    eager = make_model(execution_mode="eager").cuda()
    eager.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 128, 128, device="cuda")
    with torch.inference_mode():
        model(x)
        first_entry = model._inference_graph
    with torch.no_grad():
        actual = model(x)
        assert torch.equal(actual, eager(x))
        assert not actual.is_inference()
        assert model._inference_graph is not first_entry
        second_entry = model._inference_graph
    with torch.inference_mode():
        actual = model(x)
        assert torch.equal(actual, eager(x))
        assert actual.is_inference()
        assert model._inference_graph is not second_entry


@gpu
def test_replay_recaptures_after_attention_configuration_change():
    model = make_model().cuda()
    eager = make_model(execution_mode="eager").cuda()
    eager.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 128, 128, device="cuda")
    with torch.inference_mode():
        model(x)
        entry = model._inference_graph
        model.layers[0].attention.scale *= 2
        eager.layers[0].attention.scale *= 2
        assert torch.equal(model(x), eager(x))
        assert model._inference_graph is not entry


@gpu
def test_training_gradients_hooks_autocast_and_neighbor_shapes_bypass_capture():
    model = make_model().cuda()
    x = torch.randn(1, 128, 128, device="cuda")
    assert not model._can_replay_cuda_graph(x, None)
    with torch.inference_mode():
        assert model._can_replay_cuda_graph(x, None)
        assert not model._can_replay_cuda_graph(x.half(), None)
        assert not model._can_replay_cuda_graph(x[:, :127], None)
        assert not model._can_replay_cuda_graph(x.transpose(1, 2), None)
        with torch.autocast("cuda"):
            assert not model._can_replay_cuda_graph(x, None)
        calls = []
        handle = model.layers[0].norm1.register_forward_hook(lambda *args: calls.append(1))
        model(x)
        model(x)
        handle.remove()
        assert len(calls) == 2
        assert model.cuda_graph_replays == 0
        model.train()
        assert not model._can_replay_cuda_graph(x, None)
    model.eval()
    model(x).sum().backward()
    assert model.layers[0].attention.q_proj.weight.grad is not None
