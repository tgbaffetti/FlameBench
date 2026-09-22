"""Image orientation, valid-cell statistics and physical metrics."""
import numpy as np
import pytest
import torch
from DataProcessing.process4convolution import ImageGrid
from DataProcessing.prepare import convert
from DataProcessing.scaling import FeatureScaler
from torch.utils.data import DataLoader
from DataProcessing.Dataset import CompressorDataset, ForecasterDataset
from Baselines.OrderReduction.Linear.POD import POD
from Baselines.OrderReduction.DL.CAE import CAE
from Baselines.OrderReduction.DL.ViTAE import ViTAE
from Experiments.evaluation import evaluate


def test_image_mapping_matches_main(tmp_path):
    centers = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]])
    grid = ImageGrid(centers)
    raw = np.arange(3 * 2 * 7, dtype=np.float32).reshape(3, 2, 7)
    # Exact original script layout: (x, reversed z, field, time).
    original = np.zeros((2, 2, 2, 7), dtype=np.float32)
    original[[1, 0, 0], [0, 1, 0], :, :] = raw
    expected = original[:, ::-1].transpose(3, 2, 1, 0)
    actual = grid.images(raw.transpose(2, 1, 0))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(grid.cells(actual), raw.transpose(2, 1, 0))
    source, target = tmp_path / 'raw.npz', tmp_path / 'images.npy'
    np.savez(source, data=raw)
    convert(source, target, 2, block_cells=2, grid=grid)
    np.testing.assert_array_equal(np.load(target), expected)
    # Statistics over valid pixels equal those of the original cells.
    scaler = FeatureScaler(grid.mask).fit([actual[:3], actual[3:]])
    np.testing.assert_allclose(scaler.mean, raw.mean(axis=(0, 2)))
    np.testing.assert_allclose(scaler.scale, raw.std(axis=(0, 2)), rtol=1e-6)
    scaled = scaler.transform(actual)
    assert np.all(scaled[..., ~grid.mask] == 0)
    np.testing.assert_allclose(scaler.inverse(scaled), actual, atol=1e-6)


def test_reject_duplicate_pixels():
    with pytest.raises(ValueError, match='Multiple cells'):
        ImageGrid(np.zeros((2, 3)))


@pytest.mark.parametrize('kind', ['pod', 'cae', 'cvae', 'vit_ae'])
def test_image_pipeline_metrics(tmp_path, kind):
    grid = ImageGrid(np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]]), volumes=[1, 2, 3])
    indices = tmp_path / 'indices.npz'
    grid.save(indices)
    cells = np.broadcast_to(np.array([[1, 2, 3], [3, 5, 7]], dtype=np.float32), (12, 2, 3)).copy()
    cases = []
    for split in ['training', 'test']:
        data, phi = tmp_path / f'{split}.npy', tmp_path / f'{split}_phi.npy'
        frames = cells + np.arange(12, dtype=np.float32)[:, None, None] * 0.1 if split == "training" else cells
        np.save(data, grid.images(frames))
        np.save(phi, np.ones(12, dtype=np.float32))
        cases.append(dict(name=split, split=split, data=str(data), phi=str(phi), waveform='step'))
    metadata = dict(fields=['T', 'mix:Q'], cases=cases, dt=0.0005,
                    grid_indices=str(indices), initial_snapshot_is_steady=True)
    raw = CompressorDataset(metadata, 'train', validation_fraction=0.5, blocks=2)
    scaler = FeatureScaler(raw.mask).fit(DataLoader(raw, batch_size=3))
    split = dict(validation_fraction=0.5, blocks=2, scaler=scaler)
    compressors = {
        'pod': POD(rank=1, batch_size=3),
        'cae': CAE(rank=1, channels=(2, 4), padding=1, epochs=1, batch_size=3),
        'cvae': CAE(rank=1, channels=(2, 4), padding=1, epochs=1, batch_size=3, beta=1e-4),
        'vit_ae': ViTAE(rank=1, channels=(2, 4), padding=1, hidden=8, heads=2, layers=1, epochs=1, batch_size=3),
    }
    compressor = compressors[kind]
    compressor.fit(CompressorDataset(metadata, 'train', **split), validation=CompressorDataset(metadata, 'validation', **split))
    assert compressor.decode(compressor.encode(scaler.transform(grid.images(cells[:1])))).shape == (1, 2, 2, 2)
    # Identity codec isolates metric and Q integration from compression error.
    class Identity:
        def encode(self, x):
            return x.reshape(len(x), -1)
        def decode(self, z):
            x = z.reshape(len(z), 2, 2, 2).copy()
            x[..., ~grid.mask] = 999  # Invalid pixels must never affect metrics.
            return x
    class Constant:
        def predict(self, states, forcing):
            return states[:, -1]
    result = evaluate(Constant(), ForecasterDataset(metadata, 'test', Nx=0, Ni=0), Identity(), scaler, tmp_path)['test']
    assert result['mean_nrmse'] == pytest.approx(0, abs=1e-6)
    assert result['heat_release_relative_l2'] == pytest.approx(0, abs=1e-6)
    assert result['mean_ssim'] == pytest.approx(1)  # Exact forecast; invalid pixels (999) are ignored.
    q = np.load(tmp_path / 'test_Q.npz')
    np.testing.assert_allclose(q['reference'], 34)


@pytest.mark.parametrize('compressor', [CAE(rank=3, channels=(2, 4), padding=1), ViTAE(rank=3, channels=(2, 4), padding=1, hidden=8, heads=2, layers=1)])
def test_image_compressor_padding_and_checkpoint(compressor, tmp_path):
    import pickle
    import torch
    torch.set_num_threads(2)
    compressor.shape = (2, 9, 5)
    compressor.build_networks()
    x = torch.randn(2, 2, 9, 5)
    predicted = compressor.decoder(compressor.encoder(x))
    assert predicted.shape == x.shape
    loss = (predicted - x).square().mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for module in (compressor.encoder, compressor.decoder) for p in module.parameters())
    latent = compressor.encode(x.numpy())
    expected = compressor.decode(latent)
    path = tmp_path / 'compressor.pkl'
    path.write_bytes(pickle.dumps(compressor))
    restored = pickle.loads(path.read_bytes())
    np.testing.assert_array_equal(restored.decode(restored.encode(x.numpy())), expected)


def image_metadata(tmp_path):
    import json
    grid = ImageGrid(np.array([[1, 0, 0], [0, 0, 1], [0, 0, 0]]))
    grid.save(tmp_path / 'indices.npz')
    rng = np.random.default_rng(0)
    cases = []
    for split in ['training', 'test']:
        data, phi = tmp_path / f'{split}.npy', tmp_path / f'{split}_phi.npy'
        np.save(data, grid.images(rng.normal(size=(16, 2, 3)).astype(np.float32)))
        np.save(phi, 1 + 0.1 * np.sin(np.arange(16, dtype=np.float32)))
        cases.append(dict(name=split, split=split, data=str(data), phi=str(phi), waveform='step'))
    path = tmp_path / 'metadata.json'
    path.write_text(json.dumps(dict(fields=['T', 'mix:Q'], cases=cases, dt=0.0005,
                                    grid_indices=str(tmp_path / 'indices.npz'), initial_snapshot_is_steady=True)))
    return path


def test_joint_training_updates_compressor(tmp_path):
    import pickle
    from Experiments.run import fit, test
    config = {'metadata': str(image_metadata(tmp_path)), 'output': str(tmp_path / 'runs'), 'run_name': 'cae_gru',
              'validation_fraction': 0.5, 'blocks': 2, 'K_eval': 3, 'batch_size': 4, 'joint_batch_size': 2,
              'compressor': {'name': 'cae', 'rank': 2, 'channels': [2, 4], 'padding': 1, 'epochs': 1, 'batch_size': 4},
              'dataset': {'Nx': 1, 'Ni': 0, 'horizon': 3},
              'forecaster': {'name': 'gru', 'hiddens': [4], 'epochs': 1, 'joint_epochs': 2, 'patience': 5},
              'logging': {'wandb': {'mode': 'disabled'}}, 'evaluation': {'heat_release': False}}
    assert np.isfinite(fit(config))
    directory = tmp_path / 'runs' / 'cae_gru' / 'seed_42'
    _, frozen = pickle.loads((directory / 'preprocessing.pkl').read_bytes())
    _, tuned, _ = pickle.loads((directory / 'model.pkl').read_bytes())
    changed = [not torch.equal(a, b) for a, b in zip(frozen.encoder.state_dict().values(),
                                                    tuned.encoder.state_dict().values())]
    assert any(changed)
    rows = (directory / 'metrics.jsonl').read_text()
    assert 'joint/validation_field_loss' in rows and 'validation/field_mse' in rows
    assert set(test(config)) == {'test'}


def hpo_config(tmp_path):
    # Small fixed values keep the tiny images and 8-frame segments valid; the rest is tuned.
    return {'metadata': str(image_metadata(tmp_path)), 'validation_fraction': 0.5, 'blocks': 2, 'K_eval': 3,
            'batch_size': 4, 'joint_batch_size': 2, 'trials': 2,
            'compressor': {'rank': 2, 'epochs': 1, 'levels': 1, 'base_channels': 2, 'kernel_size': 3, 'padding': 1},
            'dataset': {'Nx': 1, 'Ni': 0, 'horizon': 3},
            'forecaster': {'hiddens': [4], 'epochs': 1, 'patience': 5}}


def test_hpo_three_stages(tmp_path):
    from Experiments.HPO import optimize
    from Baselines.Forecast.DL.networks import GRU
    best = optimize(hpo_config(tmp_path), CAE, GRU)
    assert best['compressor']['name'] == 'cae' and {'lr', 'batch_size'} <= set(best['compressor'])
    assert best['dataset'] == {'Nx': 1, 'Ni': 0, 'horizon': 3}
    assert {'name', 'lr', 'dropout', 'rollout_weight', 'joint_lr', 'joint_epochs'} <= set(best['forecaster'])
    compressor = CAE.build({k: v for k, v in best['compressor'].items() if k != 'name'})
    assert compressor.channels == (2,)


@pytest.mark.parametrize('fine_tune', [True, False])
def test_hpo_with_trained_compressor(tmp_path, fine_tune):
    from DataProcessing.metadata import load_metadata
    from Experiments.HPO import optimize
    from Baselines.Forecast.Classical.ARX import ARX
    from Baselines.Forecast.DL.networks import GRU
    config = hpo_config(tmp_path)
    metadata = load_metadata(config['metadata'])
    raw = CompressorDataset(metadata, 'train', validation_fraction=0.5, blocks=2)
    scaler = FeatureScaler(raw.mask).fit(DataLoader(raw, batch_size=4))
    split = dict(validation_fraction=0.5, blocks=2, scaler=scaler)
    compressor = CAE(rank=2, channels=(2,), padding=1, epochs=1, batch_size=4).fit(
        CompressorDataset(metadata, 'train', **split), validation=CompressorDataset(metadata, 'validation', **split))
    best = optimize(config, forecaster_class=GRU, compressor=compressor, fine_tune_compressor=fine_tune)
    assert 'compressor' not in best
    assert (best['forecaster']['joint_epochs'] > 0) == fine_tune
    # ARX overrides the dataset ranges: horizon is fixed to 1.
    config['dataset'], config['forecaster'] = {'Nx': 1, 'Ni': 0}, {}
    assert optimize(config, forecaster_class=ARX, compressor=compressor)['dataset']['horizon'] == 1


@pytest.mark.parametrize('loss', ['mse', 'huber'])
def test_compressor_loss_option_multiple_epochs(tmp_path, loss):
    from DataProcessing.metadata import load_metadata
    metadata = load_metadata(image_metadata(tmp_path))
    raw = CompressorDataset(metadata, 'train', validation_fraction=0.5, blocks=2)
    scaler = FeatureScaler(raw.mask).fit(DataLoader(raw, batch_size=4))
    split = dict(validation_fraction=0.5, blocks=2, scaler=scaler)
    training = CompressorDataset(metadata, 'train', **split)
    compressor = CAE(rank=1, padding=1, epochs=2, batch_size=4, loss=loss, beta=1e-4).fit(
        training, validation=CompressorDataset(metadata, 'validation', **split))
    assert np.isfinite(compressor.encode(np.stack([training[0].numpy(), training[1].numpy()]))).all()
    with pytest.raises(ValueError, match='loss'):
        CAE(loss='l3')


@pytest.mark.parametrize('shape', [(11, 206, 104), (2, 7, 5)])
@pytest.mark.parametrize('channels', [(4,), (4, 8, 16)])
@pytest.mark.parametrize('kernel_size, stride, padding', [(3, 2, 1), (4, 2, 1), (5, 3, 0)])
def test_cae_shapes_any_depth(shape, channels, kernel_size, stride, padding):
    compressor = CAE(rank=3, channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
    compressor.shape = shape
    if shape[1] < 100 and len(channels) == 3 and kernel_size > 3:
        with pytest.raises(ValueError, match='too small'):
            compressor.build_networks()
        return
    compressor.build_networks()
    x = torch.zeros(2, *shape)
    assert compressor.encoder(x).shape == (2, 3)
    assert compressor.decoder(torch.zeros(2, 3)).shape == (2, *shape)


@pytest.mark.parametrize('shape', [(11, 206, 104), (2, 7, 5)])
def test_vit_shapes(shape):
    compressor = ViTAE(rank=3, channels=(2, 4), padding=1, hidden=8, heads=2, layers=1)
    compressor.shape = shape
    compressor.build_networks()
    assert compressor.encoder(torch.zeros(2, *shape)).shape == (2, 3)
    assert compressor.decoder(torch.zeros(2, 3)).shape == (2, *shape)


@pytest.mark.parametrize('kind', ['gru', 'lstm', 'cnn', 'transformer'])
def test_forecaster_architecture_options(kind):
    from torch import nn
    from Baselines.Forecast.DL.networks import Recurrent, Transformer, CNN
    from Experiments.run import FORECASTERS
    options = {'gru': dict(hiddens=[8, 6], bidirectional=True, normalization='layer', activation='tanh'),
               'lstm': dict(hiddens=[8, 6, 4], normalization='batch', input_normalization='batch'),
               'cnn': dict(channels=[8, 6], kernel_size=2, normalization='batch', activation='gelu'),
               'transformer': dict(hidden=8, layers=3, heads=2, feedforward=24, activation='relu')}[kind]
    model = FORECASTERS[kind](input_size=4, output_size=3, Nx=2, Ni=1, dropout=0.1, **options)
    assert model.predict(np.zeros((2, 3, 3)), np.ones((2, 3))).shape == (2, 3)
    layers, head = model.network.layers, model.network.layers[-1]
    assert 'Linear' in str(model.network)  # Printing lists every layer.
    assert all(module.p == 0.1 for module in layers if isinstance(module, nn.Dropout))
    if kind == 'gru':
        assert isinstance(layers[0].norm, nn.Identity)  # No input normalization by default.
        assert layers[1].layer.bidirectional and isinstance(layers[2].norm, nn.LayerNorm)
        assert isinstance(layers[3], nn.Tanh) and head.in_features == 2 * 6
    if kind == 'lstm':
        assert [m.layer.hidden_size for m in layers if isinstance(m, Recurrent)] == [8, 6, 4]
        assert isinstance(layers[0].norm, nn.BatchNorm1d)  # Input normalization of each column.
        assert isinstance(layers[1].layer, nn.LSTM) and isinstance(layers[2].norm, nn.BatchNorm1d)
    if kind == 'cnn':
        assert head.in_features == (3 - 2) * 6  # Valid padding: two kernels of 2 remove two rows.
        with pytest.raises(ValueError, match='too short'):
            CNN(input_size=4, output_size=3, Nx=2, channels=[8, 8], kernel_size=3)
    if kind == 'transformer':
        encoder = layers[3]
        assert encoder.num_layers == 3 and encoder.layers[0].linear1.out_features == 24
        with pytest.raises(AssertionError, match='divisible'):
            Transformer(input_size=4, output_size=3, hidden=10, heads=4)


def test_forecaster_optimizer():
    from Baselines.Forecast.DL.networks import GRU
    model = GRU(input_size=3, output_size=2, optimizer='sgd', weight_decay=0.1)
    optimizer = model.make_optimizer(model.network.parameters(), 0.01)
    assert isinstance(optimizer, torch.optim.SGD) and optimizer.defaults['weight_decay'] == 0.1
    with pytest.raises(ValueError, match='optimizer'):
        GRU(input_size=3, output_size=2, optimizer='rmsprop')
