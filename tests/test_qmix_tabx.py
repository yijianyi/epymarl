"""TABX mixer initialization, monotonicity and checkpoint compatibility."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from modules.mixers.qmix import QMixer
from tabx_scheduler import training_config


def mixer_args(**overrides):
    config, _ = training_config([])
    config.update(n_agents=10, state_shape=340, **overrides)
    return SimpleNamespace(**config)


def test_recommended_configuration_reaches_batch_training():
    args = mixer_args()
    expected = dict(t_max=800000, lr=0.00005, gamma=0.99, grad_norm_clip=10,
                    batch_size_run=1, batch_size=32, buffer_size=256, buffer_cpu_only=True,
                    use_rnn=True, hidden_dim=128, mixing_embed_dim=64, hypernet_layers=2,
                    hypernet_embed=128, double_q=True, target_update_interval_or_tau=200,
                    epsilon_start=1.0, epsilon_finish=0.05, epsilon_anneal_time=80000,
                    evaluation_epsilon=0.0, standardise_rewards=False, standardise_returns=False,
                    mixer_init_scale=0.001, mixer_layer_norm=True, mixer_layer_norm_eps=1e-6)
    assert {key: getattr(args, key) for key in expected} == expected


@pytest.mark.parametrize("gain", [0.001, 0.002])
def test_all_mixer_linear_layers_are_scaled_orthogonal_with_zero_bias(gain):
    mixer = QMixer(mixer_args(mixer_init_scale=gain))
    linear_layers = [layer for layer in mixer.modules() if isinstance(layer, nn.Linear)]
    assert len(linear_layers) == 7
    for layer in linear_layers:
        weight = layer.weight.detach().double()
        if weight.shape[0] > weight.shape[1]:
            weight = weight.T
        gram = weight @ weight.T
        torch.testing.assert_close(gram, torch.eye(len(gram), dtype=gram.dtype) * gain**2,
                                   rtol=2e-5, atol=1e-12)
        assert torch.count_nonzero(layer.bias) == 0
    for branch in (mixer.hyper_w_1, mixer.hyper_w_final, mixer.V):
        assert [type(layer) for layer in branch] == [nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear]
        assert branch[1].eps == 1e-6
        torch.testing.assert_close(branch[1].weight, torch.ones_like(branch[1].weight))
        torch.testing.assert_close(branch[1].bias, torch.zeros_like(branch[1].bias))


def test_large_state_inputs_have_finite_small_initial_q_and_monotone_gradients():
    torch.manual_seed(7)
    mixer = QMixer(mixer_args())
    qs = (torch.randn(2, 3, 10) * 5).requires_grad_()
    states = torch.randn(2, 3, 340) * 10000
    mixed = mixer(qs, states)
    assert mixed.shape == (2, 3, 1)
    assert torch.isfinite(mixed).all() and mixed.abs().max() < 1
    mixed.sum().backward()
    assert torch.isfinite(qs.grad).all() and (qs.grad >= 0).all() and (qs.grad > 0).any()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in mixer.parameters())
    with torch.no_grad():
        for agent in range(10):
            increased = qs.detach().clone()
            increased[:, :, agent] += 1
            assert (mixer(increased, states) >= mixed.detach() - 1e-7).all()


@pytest.mark.parametrize("hypernet_layers", [1, 2])
@pytest.mark.parametrize("tabx_settings", [False, True])
def test_configured_mixer_checkpoint_roundtrip(tmp_path, hypernet_layers, tabx_settings):
    args = mixer_args(hypernet_layers=hypernet_layers)
    if not tabx_settings:
        for key in ("mixer_init_scale", "mixer_layer_norm", "mixer_layer_norm_eps"):
            delattr(args, key)
    torch.manual_seed(11)
    original = QMixer(args)
    qs, states = torch.randn(2, 3, 10), torch.randn(2, 3, 340)
    checkpoint = tmp_path / "mixer.th"
    torch.save(original.state_dict(), checkpoint)
    restored = QMixer(args)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(qs, states), original(qs, states), rtol=0, atol=0)
    norms = [layer for layer in restored.modules() if isinstance(layer, nn.LayerNorm)]
    assert len(norms) == (3 if hypernet_layers == 2 else 1) * int(tabx_settings)
    if not tabx_settings:
        # Historical configs retain the original Linear/ReLU parameter indices.
        for branch in (restored.hyper_w_1, restored.hyper_w_final, restored.V):
            if isinstance(branch, nn.Sequential):
                assert [type(layer) for layer in branch] == [nn.Linear, nn.ReLU, nn.Linear]
