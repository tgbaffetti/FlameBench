import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from DataProcessing.prepare import convert
from DataProcessing.process4convolution import ImageGrid
from DataProcessing.Dataset import CompressorDataset, ForecasterDataset
from DataProcessing.scaling import FeatureScaler
from Baselines.OrderReduction.Linear.POD import POD
from Baselines.Forecast.Classical.ARX import ARX
from Baselines.Forecast.DL.networks import GRU, LSTM, CNN, Transformer
from Experiments.metrics import FieldMetrics, gain_phase, relative_l2
from Experiments.evaluation import evaluate
from Experiments.run import fit, test as run_test, hpo
from Experiments.paths import run_directory


# Four cells on a 3 x 2 image; the other two pixels are invalid.
GRID = ImageGrid(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 2]]), volumes=[1., 2., 3., 4.])
# The 81-frame fixture needs 10 blocks of 8 frames (the default 20 would give 4 frames).
SETTINGS = {'validation_fraction': 0.2, 'blocks': 10, 'K_eval': 3, 'logging': {'wandb': {'mode': 'disabled'}}}


@pytest.fixture
def metadata(tmp_path):
    cases = []
    for name, split, amplitude in [('sweep02','training',0.2), ('sweep04','training',0.4), ('sine','test',0.3)]:
        nt = 81
        phi = 1 + amplitude*np.sin(np.arange(nt)*0.3)
        state = np.zeros(nt)
        for k in range(1,nt):
            state[k] = 0.8*state[k-1] + 0.1*(phi[k-1]-1) + 0.2*(phi[k]-1)
        spatial = np.array([[1,2,3,4], [2,4,3,1]],dtype=np.float32)
        data = 10 + state[:,None,None]*spatial + spatial[None]
        x_path, u_path = tmp_path/f'{name}.npy', tmp_path/f'phi_{name}.npy'
        np.save(x_path,GRID.images(data.astype(np.float32)));np.save(u_path,phi.astype(np.float32))
        cases.append({'name':name,'split':split,'waveform':'step','duration':0.04,
                      'data':str(x_path),'phi':str(u_path)})
    GRID.save(tmp_path/'grid.npz')
    return {'dt':0.0005,'fields':['T','mix:Q'],'initial_snapshot_is_steady':True,
            'grid_indices':str(tmp_path/'grid.npz'),'cases':cases}


def small(cls, size, **kwargs):
    """A one-layer forecaster of width 8 for states of `size` (the CNN kernel 1 fits any window)."""
    width = {'hidden': 8, 'layers': 1} if cls is Transformer else {'channels': [8], 'kernel_size': 1} if cls is CNN \
        else {'hiddens': [8]}
    return cls(input_size=size + 1, output_size=size, **width, **kwargs)


def windows(states, forcing, target):
    return [{'states': x, 'forcing': u, 'target': y} for x, u, y in zip(states, forcing, target)]


def scaled_pod(metadata, partition='train'):
    # 10 blocks of 8 frames: the 81-frame fixture is too short for the default 20 blocks.
    frames = CompressorDataset(metadata, partition, blocks=10)
    scaler = FeatureScaler(frames.mask).fit(DataLoader(frames, batch_size=8))
    return scaler, POD(rank=2, batch_size=8).fit(CompressorDataset(metadata, partition, blocks=10, scaler=scaler))


def test_stream_conversion(tmp_path):
    raw=np.arange(5*2*13,dtype=np.float64).reshape(5,2,13)
    source,target=tmp_path/'raw.npz',tmp_path/'prepared.npy'
    np.savez_compressed(source,data=raw)
    convert(source,target,2,block_cells=2)
    result=np.load(target,mmap_mode='r')
    assert isinstance(result,np.memmap)
    np.testing.assert_array_equal(result,raw.transpose(2,1,0))


def test_block_split_without_leakage(metadata):
    from DataProcessing.Dataset import split_segments
    assert split_segments(100, 'validation', 0.2, 10) == [(20, 30), (70, 80)]
    assert split_segments(100, 'train', 0.2, 10) == [(0, 20), (30, 70), (80, 100)]
    with pytest.raises(ValueError, match='validation_fraction'):
        split_segments(100, 'train', 0.01, 10)
    train = CompressorDataset(metadata, 'train', validation_fraction=0.25)
    val = CompressorDataset(metadata, 'validation', validation_fraction=0.25)
    for case in metadata['cases'][:2]:
        # Every frame belongs to exactly one partition.
        used = sorted(t for c, lo, hi in train.segments + val.segments if c is case for t in range(lo, hi))
        assert used == list(range(81))
    # The forecaster windows use the same frames as the compressor and never cross a segment.
    forecaster = ForecasterDataset(metadata, 'train', Nx=2, validation_fraction=0.25)
    assert forecaster.segments == train.segments
    assert len(forecaster) == sum(hi - lo - 3 for _, lo, hi in train.segments)
    scaler = FeatureScaler(train.mask).fit(DataLoader(train, batch_size=7))
    full = np.stack([train[i].numpy() for i in range(len(train))])[..., train.mask]
    np.testing.assert_allclose(scaler.mean, full.astype('float64').mean(axis=(0, 2)), rtol=1e-6)
    # Changing validation values must not change fitted statistics.
    for case, lo, hi in val.segments:
        a = np.load(case['data'], mmap_mode='r+'); a[lo:hi] = 1e6; a.flush()
    other = FeatureScaler(train.mask).fit(DataLoader(CompressorDataset(metadata, 'train', validation_fraction=0.25), batch_size=5))
    np.testing.assert_allclose(other.mean, scaler.mean)


def test_pod_arx_and_recursive_metrics(metadata,tmp_path):
    # The synthetic system is first order in x and uses phi(t) and phi(t+1), so Nx=0, Ni=1
    # recovery is exact. With Nx>0 and a single-frequency sine, lagged states are nearly
    # collinear and the fit is ill-posed.
    scaler,pod=scaled_pod(metadata)
    latent=ForecasterDataset(metadata,'train',Nx=0,Ni=1,scaler=scaler,compressor=pod)
    model=ARX(Nx=0,Ni=1,alpha=1e-8).fit(DataLoader(latent,batch_size=8))
    results=evaluate(model,ForecasterDataset(metadata,'test',Nx=0,Ni=1),pod,scaler,tmp_path/'eval')
    case=results['sine']
    assert case['forecast_steps']==80
    assert case['mean_nrmse']<1e-5
    assert case['heat_release_relative_l2']<1e-5
    q=np.load(tmp_path/'eval'/'sine_Q.npz')
    x=np.load(metadata['cases'][-1]['data'])
    np.testing.assert_allclose(q['reference'],GRID.cells(x)[1:,1]@np.array([1,2,3,4]),rtol=1e-6)


def test_metrics_and_harmonics():
    reference=np.array([[1,2,3],[4,4,4.]])
    metrics=FieldMetrics(['a','b']);metrics.update(reference+1,reference)
    result=metrics.result()
    assert result['field_nrmse']['a']==pytest.approx(1/np.std(reference[0]))
    assert result['field_nrmse']['b'] is None
    assert relative_l2([2,4],np.array([1,2]))==pytest.approx(1)
    t=np.arange(2001)*0.0005
    phi=1+0.2*np.sin(2*np.pi*10*t)
    q=100*(1+0.4*np.sin(2*np.pi*10*t-0.4))
    pred=100*(1+0.6*np.sin(2*np.pi*10*t-0.2))
    m=gain_phase(pred,q,phi,t,10,100)
    assert m['reference_gain']==pytest.approx(2)
    assert m['relative_gain_error']==pytest.approx(0.5)
    assert m['phase_error_deg']==pytest.approx(np.degrees(0.2))


@pytest.mark.parametrize('cls',[GRU,LSTM,CNN,Transformer])
@pytest.mark.parametrize('Ni',[0,5])
@pytest.mark.parametrize('Nx',[0,2])
def test_neural_fit_resume(cls,Nx,Ni,tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    length=max(Nx,Ni)+1
    states=torch.randn(16,length,2)
    data=DataLoader(windows(states,torch.zeros(16,length),states[:,-1:]*0.9),batch_size=8)
    model=small(cls,2,Nx=Nx,Ni=Ni,epochs=2)
    model.fit(data,data,directory=tmp_path)
    output=model.predict(states[:2].numpy(),np.zeros((2,length),dtype=np.float32))
    assert output.shape==(2,2) and np.isfinite(output).all()
    resumed=small(cls,2,Nx=Nx,Ni=Ni,epochs=3)
    resumed.fit(data,data,directory=tmp_path,resume=True)
    assert torch.load(tmp_path/'last.pt',weights_only=False)['epoch']==2


def test_full_pipeline_and_hpo(metadata,tmp_path):
    path=tmp_path/'metadata.json';path.write_text(json.dumps(metadata))
    cfg={'metadata':str(path),'output':str(tmp_path/'run'),'run_name':'pod_arx_test','seed':42,**SETTINGS,
         'compressor':{'name':'pod','rank':2,'batch_size':8},'dataset':{'Nx':2,'Ni':1,'horizon':1},
         'forecaster':{'name':'arx','alpha':1e-6},'evaluation':{'heat_release':True}}
    score=fit(cfg)
    assert np.isfinite(score)
    assert run_test(cfg)['sine']['mean_nrmse']<0.01
    assert list((run_directory(cfg)/'tensorboard').glob('events.*'))
    with pytest.raises(FileExistsError): fit(cfg)
    # Only alpha is tuned: Nx and Ni are fixed here (the fixture blocks are short) and ARX fixes horizon 1.
    cfg.update(output=str(tmp_path/'hpo'),trials=2,dataset={'Nx':2,'Ni':1},forecaster={'name':'arx'})
    best,results=hpo(cfg)
    assert best['dataset']=={'Nx':2,'Ni':1,'horizon':1}
    assert set(best['forecaster'])=={'name','alpha'} and best['compressor']=={'name':'pod','rank':2,'batch_size':8}
    assert json.loads((run_directory(best)/'config.json').read_text())==best
    assert np.isfinite(results['sine']['mean_nrmse'])


def test_physical_metadata_required(metadata,tmp_path):
    metadata['initial_snapshot_is_steady']=None
    with pytest.raises(ValueError,match='initial_snapshot'):
        evaluate(None,ForecasterDataset(metadata,'test',Nx=2),None,None,tmp_path)
    metadata['initial_snapshot_is_steady']=True
    ImageGrid(np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 2]])).save(metadata['grid_indices'])
    with pytest.raises(ValueError,match='physical cell volumes'):
        evaluate(None,ForecasterDataset(metadata,'test',Nx=2),None,None,tmp_path)


def test_reproducible_resume(tmp_path):
    # A resumed shuffled neural fit must match the uninterrupted training path.
    from utils import seed_everything
    torch.set_num_threads(1)
    states=torch.arange(72,dtype=torch.float32).reshape(12,3,2)/72
    data=windows(states,torch.zeros(12,3),states[:,-1:]*0.9)
    def loaders():
        return DataLoader(data,batch_size=4,shuffle=True),DataLoader(data,batch_size=4)
    seed_everything(7)
    full=small(GRU,2,Nx=2,Ni=0,epochs=3)
    full.fit(*loaders())
    seed_everything(7)
    part=small(GRU,2,Nx=2,Ni=0,epochs=1)
    part.fit(*loaders(),directory=tmp_path)
    continued=small(GRU,2,Nx=2,Ni=0,epochs=3)
    continued.fit(*loaders(),directory=tmp_path,resume=True)
    for key,value in full.network.state_dict().items():
        torch.testing.assert_close(value,continued.network.state_dict()[key],rtol=0,atol=0)


def test_dataloader_workers(metadata):
    dataset=ForecasterDataset(metadata,'train',Nx=2,Ni=5)
    batch=next(iter(DataLoader(dataset,batch_size=4,num_workers=2)))
    assert batch['states'].shape==(4,6,2,3,2)
    assert batch['forcing'].shape==(4,6)
    torch.testing.assert_close(batch['target'][0],dataset[0]['target'])


def test_dotenv_wandb_settings_and_secret_not_logged(tmp_path,monkeypatch):
    import os
    import wandb
    from unittest.mock import Mock
    from Experiments.logging import ExperimentLogger
    monkeypatch.chdir(tmp_path)
    for key in ['WANDB_API_KEY','WANDB_ENTITY','WANDB_PROJECT','WANDB_MODE']:
        monkeypatch.delenv(key,raising=False)
    (tmp_path/'.env').write_text('WANDB_API_KEY=test-key-not-real\nWANDB_ENTITY=FireMark\nWANDB_MODE=online\n')
    init=Mock(return_value=Mock())
    monkeypatch.setattr(wandb,'init',init)
    logger=ExperimentLogger(tmp_path/'run',{'logging':{'wandb':{'mode':'offline'}}})
    logger.log({'train/loss':0.2},0)
    logger.log({'train/loss':0.1},1)
    logger.close()
    assert os.environ['WANDB_API_KEY']=='test-key-not-real'
    assert init.call_args.kwargs['entity']=='FireMark'
    assert init.call_args.kwargs['mode']=='online'
    assert 'test-key-not-real' not in json.dumps(init.call_args.kwargs['config'])
    assert 'test-key-not-real' not in (tmp_path/'run'/'metrics.jsonl').read_text()
    records=[json.loads(line) for line in (tmp_path/'run'/'metrics.jsonl').read_text().splitlines()]
    assert [record['step'] for record in records]==[0,1]


def test_walkthrough_notebook(metadata,tmp_path,monkeypatch):
    notebook_path=Path(__file__).resolve().parents[1]/'Walkthrough.ipynb'
    notebook=json.loads(notebook_path.read_text())
    assert notebook['nbformat']==4
    headings=[''.join(c['source']).splitlines()[0] for c in notebook['cells']
              if c['cell_type']=='markdown' and ''.join(c['source']).startswith('## ')]
    assert headings==['## 1. Data preparation','## 2. Datasets','## 3. POD-ARX','## 4. POD-LSTM','## 5. CAE-ARX',
                      '## 6. HPO of POD-ARX','## 7. Results']
    for case in metadata['cases']:
        case['source_data']=case['name']+'.npz'
        case['source_phi']='phi_'+case['name']+'.npz'
    metadata_path=tmp_path/'metadata.json'
    metadata_path.write_text(json.dumps(metadata))
    monkeypatch.chdir(tmp_path)
    namespace={}
    first=True
    for index,cell in enumerate(notebook['cells']):
        if cell['cell_type']!='code':
            continue
        source=''.join(cell['source'])
        exec(compile(source,f'Walkthrough.ipynb:cell{index}','exec'),namespace)
        if first:
            # Only change data paths, small-example sizes, and the logging backend.
            namespace.update(metadata_path=metadata_path,raw_directory=tmp_path/'Raw',
                             Nx=2,Ni=4,horizon=2,K_eval=3,rank=2,batch_size=8,blocks=10,
                             lstm_epochs=1,cae_epochs=1,cae_channels=[2],hpo_fixed_dataset={'Nx':2,'Ni':4},
                             logging_config={'logging':{'wandb':{'mode':'disabled'}}})
            first=False
    # This is a notebook execution check, not a claim about accuracy.
    assert set(namespace['tests'])=={'pod_arx','pod_lstm','cae_arx','hpo_pod_arx'}
    for results in namespace['tests'].values():
        result=results['sine']
        assert np.isfinite(result['mean_nrmse'])
        assert result['heat_release_status']=='explicitly_disabled'
        assert result['first_predicted_index']==1
    assert namespace['best']['dataset']['horizon']==1
    assert (namespace['output']/'hpo_pod_arx'/'seed_42'/'model.pkl').exists()


@pytest.mark.parametrize('Nx,Ni,horizon',[(0,0,1),(4,1,1),(1,5,1),(2,1,4)])
def test_forecaster_windows(metadata,Nx,Ni,horizon):
    length=max(Nx,Ni)+1
    for partition in ['train','validation']:
        scaler,pod=scaled_pod(metadata,partition)
        images=ForecasterDataset(metadata,partition,Nx=Nx,Ni=Ni,horizon=horizon,blocks=10)
        latent=ForecasterDataset(metadata,partition,Nx=Nx,Ni=Ni,horizon=horizon,blocks=10,scaler=scaler,compressor=pod)
        assert len(images)==len(latent)==sum(hi-lo-length-horizon+1 for _,lo,hi in images.segments)
        offset=0
        for case,lo,hi in images.segments:
            fields,phi=images.arrays(case)
            for t in [lo+length-1,hi-horizon-1]:  # First and last window of the segment.
                index=offset+t-(lo+length-1)
                states=fields[t-length+1:t+1].copy()
                states[:length-1-Nx]=0
                forcing=phi[t-length+2:t+horizon+1]-1
                forcing[:length-1-Ni]=0
                sample=images[index]
                np.testing.assert_array_equal(sample['states'],states)
                np.testing.assert_allclose(sample['forcing'],forcing)
                np.testing.assert_array_equal(sample['target'],fields[t+1:t+horizon+1])
                assert t-length+1>=lo and t+horizon<hi
                encoded=pod.encode(scaler.transform(fields[t-length+1:t+horizon+1].copy()))
                encoded[:length-1-Nx]=0
                np.testing.assert_allclose(latent[index]['states'],encoded[:length],atol=1e-5)
                np.testing.assert_allclose(latent[index]['target'],encoded[length:],atol=1e-5)
            offset+=hi-lo-length-horizon+1


def test_forcing_padding_in_rollouts(metadata,tmp_path):
    from Experiments.evaluation import forcing_window, validation_error
    # Predicting x(1) with 5 rows reads phi(-3..1) - 1: zero before time zero, and zero for
    # rows older than the last Ni + 1.
    np.testing.assert_allclose(forcing_window(np.array([1.3,1.3,1.3]),1,5,4),[[0,0,0,0.3,0.3]],atol=1e-6)
    np.testing.assert_allclose(forcing_window(np.array([1.3,1.3,1.3]),1,5,0),[[0,0,0,0,0.3]],atol=1e-6)
    scaler,pod=scaled_pod(metadata)
    class Recorder:
        def __init__(self):
            self.inputs=[]
        def predict(self,states,forcing):
            self.inputs.append((states.copy(),forcing.copy()))
            return states[:,-1]
    recorder=Recorder()
    test_data=ForecasterDataset(metadata,'test',Nx=0,Ni=4)
    evaluate(recorder,test_data,pod,scaler,tmp_path/'eval',heat_release=False)
    phi=np.load(metadata['cases'][-1]['phi'])
    for k,(states,u) in enumerate(recorder.inputs[1:],start=1):
        assert states.shape==(1,5,2) and not states[:,:4].any()
        expected=[0 if t<0 else phi[t]-1 for t in range(k-4,k+1)]
        np.testing.assert_allclose(u[0],expected,atol=1e-6)
    observed=Recorder()
    result=evaluate(observed,test_data,pod,scaler,tmp_path/'observed',
                    heat_release=False,initialization='observed_history')
    assert result['sine']['first_predicted_index']==5
    fields=np.load(metadata['cases'][-1]['data'])
    np.testing.assert_allclose(observed.inputs[0][0][0,-1],pod.encode(scaler.transform(fields[4:5]))[0],atol=1e-6)
    np.testing.assert_allclose(observed.inputs[0][1][0],phi[1:6]-1,atol=1e-6)
    validation=ForecasterDataset(metadata,'validation',Nx=0,Ni=4,horizon=3,stride=3,blocks=10,scaler=scaler)
    recorder=Recorder()
    validation_error(recorder,pod,validation)
    case,lo,_=validation.segments[0]
    np.testing.assert_allclose(recorder.inputs[0][1][0],np.load(case['phi'])[lo+1:lo+6]-1,atol=1e-6)
    latent_validation=ForecasterDataset(metadata,'validation',Nx=0,Ni=4,horizon=3,stride=3,
                                        blocks=10,scaler=scaler,compressor=pod)
    recorder=Recorder()
    assert np.isfinite(validation_error(recorder,pod,latent_validation))
    np.testing.assert_allclose(recorder.inputs[0][1][0],np.load(case['phi'])[lo+1:lo+6]-1,atol=1e-6)


@pytest.mark.parametrize('Nx,Ni',[(-1,0),(0,-1),(1.5,0),(0,True)])
def test_invalid_histories(metadata,Nx,Ni):
    with pytest.raises(ValueError,match='nonnegative integers'):
        ForecasterDataset(metadata,'train',Nx=Nx,Ni=Ni)


@pytest.mark.parametrize('cls',[GRU,LSTM,CNN,Transformer])
@pytest.mark.parametrize('Nx,Ni',[(0,0),(4,1),(1,5)])
def test_forcing_enters_neural_core(cls,Nx,Ni):
    torch.manual_seed(19)
    model=small(cls,2,Nx=Nx,Ni=Ni)
    network=model.network
    network.eval()
    length=max(Nx,Ni)+1
    states=torch.randn(2,length,2)
    forcing=torch.randn(2,length,requires_grad=True)
    core_outputs=[]
    def capture_core(module,args,output):
        core_outputs.append(output)
    # The layers before the last-row readout and the linear head.
    hook=network.layers[-3].register_forward_hook(capture_core)
    prediction=network(states,forcing)
    # Every supplied forcing value must affect the core, before the prediction head.
    # Random weights: a plain sum of layer-normalized features is constant.
    core=core_outputs[0]
    core_gradient=torch.autograd.grad((core*torch.randn_like(core)).sum(),forcing,retain_graph=True)[0]
    assert torch.isfinite(core_gradient).all()
    assert (core_gradient.abs().sum(dim=0)>0).all()
    prediction.sum().backward()
    assert (forcing.grad.abs().sum(dim=0)>0).all()
    with torch.no_grad():
        changed=network(states,forcing.detach()+0.5)
    assert not torch.allclose(prediction,changed)
    hook.remove()


def test_multi_seed_cli_layout(metadata,tmp_path,monkeypatch):
    import sys
    from Experiments.run import main
    metadata_path=tmp_path/'metadata.json'
    metadata_path.write_text(json.dumps(metadata))
    cfg={'metadata':str(metadata_path),'output':str(tmp_path/'results'),**SETTINGS,
         'compressor':{'name':'pod','rank':2,'batch_size':8},'dataset':{'Nx':2,'Ni':0},
         'forecaster':{'name':'arx','alpha':1e-6},'evaluation':{'heat_release':True}}
    config_path=tmp_path/'config.json'
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(sys,'argv',['flamebench','run','--config',str(config_path),'--seeds','0','1'])
    main()
    groups=list((tmp_path/'results').iterdir())
    assert len(groups)==1 and groups[0].name.startswith('pod_arx_')
    for seed in [0,1]:
        directory=groups[0]/f'seed_{seed}'
        saved=json.loads((directory/'config.json').read_text())
        assert saved['seed']==seed
        assert saved['run_name']==groups[0].name
        assert run_directory(saved)==directory
        assert (directory/'metrics.json').exists()
        assert (directory/'model.pkl').exists()
        records=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        assert any('validation/field_mse' in row for row in records)
        assert any('test/sine/mean_nrmse' in row for row in records)
        assert list((directory/'tensorboard').glob('events.*'))
        assert not (directory/'evaluation').exists()
    # Resume the named seed in place; never create another timestamp.
    seed0=json.loads((groups[0]/'seed_0'/'config.json').read_text())
    fit(seed0,resume=True)
    assert len(list((tmp_path/'results').iterdir()))==1
    with pytest.raises(FileExistsError):
        fit(seed0)


def test_run_directory_requires_explicit_group(tmp_path):
    cfg={'output':str(tmp_path),'seed':7}
    with pytest.raises(ValueError,match='run_name'):
        run_directory(cfg)
    cfg['run_name']='pod_gru_20260920T120000Z'
    assert run_directory(cfg)==tmp_path/cfg['run_name']/'seed_7'
    cfg['run_name']='../wrong'
    with pytest.raises(ValueError,match='single folder'):
        run_directory(cfg)


def test_wandb_seed_group_and_resume_id(tmp_path,monkeypatch):
    import wandb
    from unittest.mock import Mock
    from Experiments.logging import ExperimentLogger
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('WANDB_MODE','online')
    init=Mock(return_value=Mock())
    monkeypatch.setattr(wandb,'init',init)
    cfg={'run_name':'pod_gru_timestamp','seed':3,'logging':{'wandb':{'mode':'online'}}}
    directory=tmp_path/cfg['run_name']/'seed_3'
    logger=ExperimentLogger(directory,cfg)
    logger.close()
    first=init.call_args.kwargs
    assert first['group']=='pod_gru_timestamp'
    assert first['name']=='pod_gru_timestamp/seed_3'
    logger=ExperimentLogger(directory,cfg,resume=True)
    logger.close()
    assert init.call_args.kwargs['id']==first['id']
    assert init.call_args.kwargs['resume']=='allow'


def test_pod_matches_legacy(metadata, tmp_path):
    from legacy.models import PODReducer
    rng = np.random.default_rng(31)
    grid = ImageGrid(np.array([[1, 0, 0], [0, 0, 2], [0, 0, 0], [1, 0, 1]]))
    grid.save(tmp_path / 'grid.npz')
    for case in metadata['cases']:
        np.save(case['data'], grid.images(rng.normal(size=(81, 2, 4)).astype(np.float32)))
    raw = CompressorDataset(metadata, 'train')
    scaler = FeatureScaler(raw.mask).fit(DataLoader(raw, batch_size=7))
    dataset = CompressorDataset(metadata, 'train', scaler=scaler)
    scaled = np.concatenate([x.numpy() for x in DataLoader(dataset, batch_size=11)])
    vectors = grid.cells(scaled)
    legacy = PODReducer(rank=3)
    legacy.fit(vectors)
    current = POD(rank=3, batch_size=7).fit(dataset)
    np.testing.assert_allclose(current.mean, legacy.mean, atol=1e-6)
    np.testing.assert_allclose(current.U_r, legacy.U_r, atol=1e-5)
    expected_z = legacy.encode_torch(torch.from_numpy(vectors)).numpy()
    np.testing.assert_allclose(current.encode(scaled), expected_z, atol=1e-5)
    expected = legacy.decode_torch(torch.from_numpy(expected_z), vectors.shape[1:]).numpy()
    decoded = current.decode(expected_z)
    np.testing.assert_allclose(grid.cells(decoded), expected, atol=1e-5)
    assert not decoded[..., ~grid.mask].any()
    other = POD(rank=3, batch_size=13).fit(dataset)
    np.testing.assert_allclose(other.U_r, current.U_r, atol=1e-5)


def test_unknown_config_keys_raise():
    from Experiments.run import check_config
    from Baselines.Forecast.DL.networks import GRU
    from Baselines.Forecast.Classical.ARX import ARX
    from Baselines.OrderReduction.DL.CAE import CAE
    with pytest.raises(ValueError, match='batch_szie'):
        check_config({'batch_szie': 3})
    for build in (lambda: GRU(input_size=3, output_size=2, hiden=8), lambda: ARX(aplha=1.0), lambda: CAE(chanels=2)):
        with pytest.raises(TypeError):
            build()


def zero_increment_gru(**kwargs):
    # Zero increments: the prediction stays at the last state, so step errors are known exactly.
    model=GRU(input_size=2,output_size=1,Nx=0,Ni=0,hiddens=[2],rollout_weight=0.5,**kwargs)
    for p in model.network.layers[-1].parameters():
        torch.nn.init.zeros_(p)
    return model


@pytest.mark.parametrize('loss,per_step',[('mse',[1,4,9]),('mae',[1,2,3]),
                                          ('huber',[0.5,1.5,2.5]),('smooth_l1',[0.5,1.5,2.5])])
def test_rollout_loss_combines_one_and_multi_step(loss,per_step):
    model=zero_increment_gru(loss=loss)
    target=torch.tensor([[[1.],[2.],[3.]]])
    total,steps=model.rollout_loss(torch.zeros(1,1,1),torch.zeros(1,3),target)
    np.testing.assert_allclose(steps.numpy(),per_step)
    assert total.item()==pytest.approx(0.5*per_step[0]+0.5*np.mean(per_step))
    assert not steps.requires_grad
    with pytest.raises(ValueError,match='rollout_weight'):
        GRU(input_size=2,output_size=1,rollout_weight=1.5)
    with pytest.raises(ValueError,match='joint_epochs'):
        GRU(input_size=2,output_size=1,joint_epochs=-1)
    with pytest.raises(ValueError,match='loss'):
        GRU(input_size=2,output_size=1,loss='l3')


def test_detach_rollout_cuts_gradient_through_fed_back_predictions():
    # Same weights, same loss value; only the gradient path through fed-back predictions differs.
    values,gradients=[],[]
    for detach in (False,True):
        torch.manual_seed(0)
        model=GRU(input_size=2,output_size=1,Nx=1,Ni=0,hiddens=[2],detach_rollout=detach)
        states,target=torch.randn(4,2,1),torch.randn(4,3,1)
        total,_=model.rollout_loss(states,torch.zeros(4,4),target)
        total.backward()
        values.append(total.item())
        gradients.append(torch.cat([p.grad.flatten() for p in model.network.parameters()]))
    assert values[0]==pytest.approx(values[1])
    assert not torch.allclose(*gradients)


def test_validation_horizon_longer_than_training():
    # Trained on 2-step rollouts, validated on 5-step rollouts.
    torch.manual_seed(0)
    class Logger:
        def __init__(self):
            self.rows=[]
        def log(self,values,step):
            self.rows.append(values)
    train=DataLoader(windows(torch.randn(8,3,2),torch.zeros(8,4),torch.randn(8,2,2)),batch_size=4)
    val=DataLoader(windows(torch.randn(8,3,2),torch.zeros(8,7),torch.randn(8,5,2)),batch_size=4)
    logger=Logger()
    small(GRU,2,Nx=2,epochs=1).fit(train,val,logger=logger)
    keys=[key for key in logger.rows[0] if key.startswith('validation/latent_step_')]
    assert len(keys)==5
