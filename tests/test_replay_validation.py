"""One traversal preserves the prior ordered state and all replay guards."""
import copy
import pytest
import torch
from torch import nn

from benchmarks.compare_replay_validation import prior_signature
from transformer_opt.kernels import triton_available
from transformer_opt.submission import TransformerConfig, UserOptimizedTransformer


def model(**kwargs):
    return UserOptimizedTransformer(TransformerConfig(1, 128, 128, 4, 128, 4, True), **kwargs).eval()


def test_signature_matches_prior_order_with_aliases_none_and_shared_storage():
    m = model()
    x = torch.randn(1, 128, 128)
    m.register_module('optional', None)
    m.register_module('duplicate', m.layers[0].norm1)
    m.register_parameter('shared_storage', nn.Parameter(m.final_norm.bias.detach()))
    m.register_parameter('absent', None)
    assert m._graph_signature(x, None) == prior_signature(m, x, None)
    # Module iterators also deduplicate cycles; the direct traversal must not
    # hang or duplicate the state when a registry points to an ancestor.
    m._modules['cycle'] = m
    try:
        assert m._graph_signature(x, None) == prior_signature(m, x, None)
    finally:
        del m._modules['cycle']


def test_signature_detects_storage_identity_metadata_and_module_mutations():
    m = model()
    x = torch.randn(1, 128, 128)
    previous = m._graph_signature(x, None)
    def changed():
        nonlocal previous
        current = m._graph_signature(x, None)
        assert current == prior_signature(m, x, None)
        assert current != previous
        previous = current
    with torch.no_grad():
        m.final_norm.bias.add_(.1); changed()
        old_version = m.final_norm.bias._version
        m.final_norm.bias.data = m.final_norm.bias.data.clone()
        assert m.final_norm.bias._version == old_version
        changed()
        m.final_norm.bias = nn.Parameter(m.final_norm.bias.clone()); changed()
        m.final_norm = copy.deepcopy(m.final_norm); changed()
        m.final_norm.eps *= 2; changed()
        m.final_norm.normalized_shape = (64, 2); changed()
        m.layers[0].attention.scale *= 2; changed()
        m.half(); changed()


def test_signature_rejects_hooks_training_and_inference_parameters():
    m = model()
    x = torch.randn(1, 128, 128)
    m.layers[0].norm1.train()
    assert m._graph_signature(x, None) is None
    m.eval()
    for register in (m.layers[0].norm1.register_forward_hook, m.layers[0].norm1.register_forward_pre_hook):
        hook = register(lambda *args: None)
        try: assert m._graph_signature(x, None) is None
        finally: hook.remove()
    with torch.inference_mode():
        m.final_norm.bias = nn.Parameter(m.final_norm.bias.clone())
    assert m._graph_signature(x, None) is None


@pytest.mark.skipif(not triton_available(), reason='CUDA Triton unavailable')
def test_replay_recaptures_on_storage_replacement_and_module_replacement():
    m = model().cuda()
    eager = model(execution_mode='eager').cuda()
    eager.load_state_dict(m.state_dict(), strict=True)
    x = torch.randn(1, 128, 128, device='cuda')
    with torch.inference_mode():
        m(x)
        entry = m._inference_graph
        # Ordinary replacement tensors retain replay eligibility; inference
        # tensors intentionally trigger the separate fallback gate above.
        with torch.inference_mode(False), torch.no_grad():
            for item in (m, eager):
                item.final_norm.bias.data = (item.final_norm.bias.data + .1).clone()
        assert torch.equal(m(x), eager(x))
        assert m._inference_graph is not entry
        entry = m._inference_graph
        with torch.inference_mode(False):
            for item in (m, eager):
                item.layers[0].norm1 = copy.deepcopy(item.layers[0].norm1)
        assert torch.equal(m(x), eager(x))
        assert m._inference_graph is not entry
