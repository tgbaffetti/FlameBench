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
from Experiments.metrics import FieldMetrics, FieldSSIM, gain_phase, relative_l2
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
    best,results=hpo(cfg,[0,1])
    assert best['dataset']=={'Nx':2,'Ni':1,'horizon':1}
    assert set(best['forecaster'])=={'name','alpha'} and best['compressor']=={'name':'pod','rank':2,'batch_size':8}
    # POD and ARX are deterministic: every seed would give the same model, so only the first runs.
    assert set(results)=={0} and np.isfinite(results[0]['sine']['mean_nrmse'])
    saved=json.loads((tmp_path/'hpo'/'pod_arx_test'/'seed_0'/'config.json').read_text())
    assert saved=={**best,'seed':0} and saved['stage']=='fit' and saved['validation_fraction']==0.2
    assert not (tmp_path/'hpo'/'pod_arx_test'/'seed_1').exists()
    assert not (tmp_path/'hpo'/'pod_arx_test_refit').exists()


def test_hpo_fits_deterministic_compressor_once(metadata,tmp_path,monkeypatch):
    # POD is deterministic: the seeds reuse the first seed's POD; the GRU is fitted per seed.
    import pickle
    from Experiments.pipeline import Pipeline
    path=tmp_path/'metadata.json';path.write_text(json.dumps(metadata))
    cfg={'metadata':str(path),'output':str(tmp_path/'hpo'),'run_name':'pod_gru','seed':42,**SETTINGS,'trials':1,
         'compressor':{'name':'pod','rank':2,'batch_size':8},'dataset':{'Nx':2,'Ni':1,'horizon':2},
         'forecaster':{'name':'gru','hiddens':[4],'epochs':2,'patience':2,'lr':1e-3,'optimizer':'adam',
                       'weight_decay':0.0,'dropout':0.0,'rollout_weight':0.5,'normalization':None}}
    fits=[]
    original=Pipeline.train_compressor
    monkeypatch.setattr(Pipeline,'train_compressor',lambda self,*args,**kwargs: fits.append(1) or original(self,*args,**kwargs))
    best,results=hpo(cfg,[0,1])
    assert set(results)=={0,1} and len(fits)==2  # one in the HPO, one for seed 0
    saved=[pickle.loads((tmp_path/'hpo'/'pod_gru'/f'seed_{seed}'/'preprocessing.pkl').read_bytes())[1] for seed in (0,1)]
    np.testing.assert_array_equal(saved[0].U_r,saved[1].U_r)
    networks=[pickle.loads((tmp_path/'hpo'/'pod_gru'/f'seed_{seed}'/'model.pkl').read_bytes())[2].network for seed in (0,1)]
    assert any(not torch.equal(a,b) for a,b in zip(networks[0].state_dict().values(),networks[1].state_dict().values()))


def test_refit_split_uses_every_frame(metadata):
    from DataProcessing.Dataset import split_segments
    assert split_segments(81,'train',0,10)==[(0,81)] and split_segments(81,'validation',0,10)==[]
    assert len(CompressorDataset(metadata,'train',validation_fraction=0))==2*81
    with pytest.raises(ValueError,match='No cases'):
        CompressorDataset(metadata,'validation',validation_fraction=0)


def test_refit_keeps_optimizer_steps(metadata,tmp_path):
    from Experiments.run import refit, refit_config
    path=tmp_path/'metadata.json';path.write_text(json.dumps(metadata))
    cfg={'metadata':str(path),'output':str(tmp_path/'run'),'run_name':'pod_gru','seed':42,**SETTINGS,
         'compressor':{'name':'pod','rank':2,'batch_size':8},'dataset':{'Nx':2,'Ni':1,'horizon':2},
         'forecaster':{'name':'gru','hiddens':[4],'epochs':6,'patience':6}}
    fit(cfg)
    fitted=json.loads((run_directory(cfg)/'summary.json').read_text())
    best=fitted['best_epochs']['forecaster']
    assert 1<=best<=6 and fitted['train_frames']==len(CompressorDataset(metadata,'train',validation_fraction=0.2,blocks=10))
    refitted=refit_config(cfg)
    assert refitted['forecaster']['epochs']==max(1,round(best*fitted['train_frames']/(2*81)))
    assert refitted['run_name']=='pod_gru_refit' and cfg['forecaster']['epochs']==6
    refit(cfg,[0,1])
    for seed in (0,1):
        directory=tmp_path/'run'/'pod_gru_refit'/f'seed_{seed}'
        assert json.loads((directory/'summary.json').read_text())['validation_field_mse'] is None
        summary=[json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines() if 'summary' in line]
        assert np.isfinite(summary[-1]['summary']['Test_Summary/nrmse'])


def test_constant_baseline_repeats_initial_field(metadata,tmp_path):
    path=tmp_path/'metadata.json';path.write_text(json.dumps(metadata))
    cfg={'metadata':str(path),'output':str(tmp_path/'run'),'run_name':'constant','seed':42,**SETTINGS,
         'compressor':{'name':'identity'},'dataset':{'Nx':0,'Ni':0,'horizon':1},'forecaster':{'name':'constant'},
         'evaluation':{'heat_release':True}}
    assert np.isfinite(fit(cfg))
    result=run_test(cfg)['sine']
    # The prediction is the steady first frame, so the heat release error equals that of repeating it.
    frames=np.load(metadata['cases'][2]['data'])
    rows,columns=GRID.rows,GRID.columns
    q=frames[:,1][...,rows,columns]@np.array([1.,2.,3.,4.])
    assert result['heat_release_relative_l2']==pytest.approx(np.linalg.norm(q[1:]-q[0])/np.linalg.norm(q[1:]),rel=1e-5)


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
    # The notebook uses CAE's default 4x4 kernel; give this tiny fixture a valid image size.
    notebook_grid = ImageGrid(np.array([[x, 0, z] for z in range(8) for x in range(8)]), volumes=np.ones(64))
    notebook_grid.save(metadata['grid_indices'])
    for case in metadata['cases']:
        np.save(case['data'], notebook_grid.images(np.tile(GRID.cells(np.load(case['data'])), (1, 1, 16))))
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
                             loader_options={'num_workers': 0},
                             logging_config={'logging':{'wandb':{'mode':'disabled'}}})
            first=False
    # This is a notebook execution check, not a claim about accuracy.
    assert set(namespace['tests'])=={'pod_arx','pod_lstm','cae_arx','hpo_pod_arx'}
    for results in namespace['tests'].values():
        result=results['sine']
        assert np.isfinite(result['mean_nrmse'])
        assert np.isfinite(result['heat_release_relative_l2']) and 0<result['mean_ssim']<=1
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
        assert any('Validation/field_mse' in row for row in records)
        assert any('Test/sine/nrmse' in row for row in records)
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
    cfg={'run_name':'pod_gru_timestamp','seed':3,'stage':'fit','compressor':{'name':'pod'},'forecaster':{'name':'gru'},
         'logging':{'wandb':{'mode':'online','tags':['benchmark-v2']}}}
    directory=tmp_path/cfg['run_name']/'seed_3'
    logger=ExperimentLogger(directory,cfg)
    logger.summary({'Test_Summary/nrmse':0.2,'Test_Summary/ssim':None})
    first=init.call_args.kwargs
    # One W&B group per model: its HPO run and every seed.
    assert first['group']=='pod_gru' and first['config']['model_name']=='pod_gru'
    assert first['name']=='pod_gru_timestamp/seed_3' and first['job_type']=='fit'
    assert first['tags']==['benchmark-v2','fit']
    # Summaries also enter the history, where W&B builds its automatic panels; None is skipped.
    init.return_value.log.assert_called_with({'Test_Summary/nrmse':0.2})
    # A curve gets its own x-axis.
    logger.log({'Train/forecaster_loss':0.5},3,axis='Train/forecaster_epoch')
    init.return_value.define_metric.assert_called_with('Train/forecaster_loss',step_metric='Train/forecaster_epoch')
    init.return_value.log.assert_called_with({'Train/forecaster_epoch':3,'Train/forecaster_loss':0.5})
    logger.close()
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
        def log(self,values,step=0,axis=None):
            self.rows.append((values,step,axis))
    train=DataLoader(windows(torch.randn(8,3,2),torch.zeros(8,4),torch.randn(8,2,2)),batch_size=4)
    val=DataLoader(windows(torch.randn(8,3,2),torch.zeros(8,7),torch.randn(8,5,2)),batch_size=4)
    logger=Logger()
    small(GRU,2,Nx=2,epochs=2).fit(train,val,logger=logger)
    assert [(step,axis) for _,step,axis in logger.rows]==[(1,'Train/forecaster_epoch'),(2,'Train/forecaster_epoch')]
    assert all(np.isfinite(values['Validation/forecaster_loss']) for values,_,_ in logger.rows)


def test_configurable_loader_preserves_frames_and_reuses_workers(metadata, monkeypatch):
    from DataProcessing.Dataset import Dataset
    from DataProcessing.loading import make_loader
    from Experiments.pipeline import Pipeline

    monkeypatch.setattr(Dataset, 'in_memory', True)
    dataset = CompressorDataset(metadata, 'train')
    options = dict(num_workers=2, persistent_workers=True, prefetch_factor=1,
                   pin_memory=False, multiprocessing_context='spawn')
    pipeline = Pipeline({'dataloader': options}, metadata)
    loader = pipeline.loader(dataset, batch_size=7)
    assert dataset.__getstate__()['in_memory'] is False
    expected = torch.stack([dataset[i] for i in range(len(dataset))])
    first = torch.cat(list(loader))
    pids = [worker.pid for worker in loader._iterator._workers]
    second = torch.cat(list(loader))
    assert [worker.pid for worker in loader._iterator._workers] == pids
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, expected, rtol=0, atol=0)
    single = make_loader(dataset, batch_size=7, **{**options, 'num_workers': 0})
    assert single.prefetch_factor is None and not single.persistent_workers
    torch.testing.assert_close(torch.cat(list(single)), expected, rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
def test_torch_pod_reconstruction_and_rng(metadata, tmp_path, device):
    import pickle
    rng = np.random.default_rng(6)
    grid = ImageGrid(np.array([[i, 0, 0] for i in range(32)]))
    grid.save(tmp_path / 'grid.npz')
    basis = rng.normal(size=(3, 64))
    for case in metadata['cases']:
        vectors = rng.normal(size=(81, 3)) @ basis + rng.normal(size=(81, 64)) * 0.01 + 2
        np.save(case['data'], grid.images(vectors.reshape(81, 2, 32).astype(np.float32)))
    dataset = CompressorDataset(metadata, 'train')
    frames = np.stack([dataset[i].numpy() for i in range(len(dataset))])
    rng_before = torch.get_rng_state().clone()
    # A DataLoader consumes a worker seed even without workers. SVD must add no RNG side effects.
    list(DataLoader(dataset, batch_size=64))
    rng_after_loading = torch.get_rng_state().clone()
    torch.set_rng_state(rng_before)
    cuda_before = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    pod = POD(rank=3, backend='torch', device=device).fit(dataset)
    assert torch.equal(torch.get_rng_state(), rng_after_loading)
    if cuda_before is not None:
        assert torch.equal(torch.cuda.get_rng_state(), cuda_before)
    vectors = grid.cells(frames).reshape(len(frames), -1)
    centered = vectors - vectors.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    reconstruction = grid.cells(pod.decode(pod.encode(frames))).reshape(vectors.shape)
    np.testing.assert_allclose(np.linalg.norm(vectors - reconstruction) ** 2,
                               np.square(singular_values[3:]).sum(), rtol=0.002)
    np.testing.assert_allclose(pod.U_r.T @ pod.U_r, np.eye(3), atol=2e-5)
    np.testing.assert_allclose(pod.singular_values, singular_values[:3], rtol=2e-5)
    restored = pickle.loads(pickle.dumps(pod))
    np.testing.assert_array_equal(restored.decode(restored.encode(frames)), pod.decode(pod.encode(frames)))
    repeated = POD(rank=3, backend='torch', device=device).fit(dataset)
    np.testing.assert_allclose(repeated.U_r, pod.U_r, atol=1e-6)


def test_fit_progress_is_plain_text(metadata, capsys):
    dataset = CompressorDataset(metadata, 'train')
    POD(rank=2).fit(dataset)
    output = capsys.readouterr().err
    assert 'POD snapshots' in output and 'POD SVD' in output and '100%' in output


def test_field_ssim():
    rng = np.random.default_rng(0)
    mask = np.ones((20, 16), dtype=bool)
    mask[:, :3] = False
    reference = rng.normal(size=(2, 20, 16))
    ssim = FieldSSIM(['a', 'b'], mask, np.ptp(reference, axis=(1, 2)))
    ssim.update(np.where(mask, reference, 99.0), reference)  # Invalid pixels do not count.
    assert ssim.result()['mean_ssim'] == pytest.approx(1)
    noisy = FieldSSIM(['a', 'b'], mask, np.ptp(reference, axis=(1, 2)))
    noisy.update(reference + rng.normal(size=reference.shape), reference)
    assert all(value < 0.9 for value in noisy.result()['field_ssim'].values())


@pytest.mark.parametrize('shape',[(3,2),(20,13)])
def test_batched_ssim_matches_scipy(shape):
    # The torch SSIM (batched, GPU-capable) reproduces the per-frame scipy computation it replaced.
    from scipy.ndimage import gaussian_filter
    rng=np.random.default_rng(0)
    mask=rng.random(shape)>0.2
    reference=rng.normal(size=(5,3,*shape));predicted=reference+0.3*rng.normal(size=reference.shape)
    data_range=np.ptp(reference,axis=(0,2,3))
    c1,c2=((0.01*data_range)**2)[:,None,None],((0.03*data_range)**2)[:,None,None]
    blur=lambda x:gaussian_filter(x,1.5,truncate=3.5,axes=(1,2))
    expected=[]
    for p,r in zip(predicted,reference):
        p,r=np.where(mask,p,0),np.where(mask,r,0)
        mp,mr=blur(p),blur(r);vp,vr=blur(p*p)-mp**2,blur(r*r)-mr**2;cov=blur(p*r)-mp*mr
        expected.append((((2*mp*mr+c1)*(2*cov+c2))/((mp**2+mr**2+c1)*(vp+vr+c2)))[:,mask].mean(axis=1))
    ssim=FieldSSIM(['a','b','c'],mask,data_range);ssim.update_batch(predicted,reference)
    np.testing.assert_allclose(list(ssim.result()['field_ssim'].values()),np.mean(expected,axis=0),rtol=0,atol=1e-12)


def test_parallel_hpo_and_seeds(metadata,tmp_path):
    # parallel 2: trials of each stage and the seed fits run in worker processes (spawn).
    import pickle
    path=tmp_path/'metadata.json';path.write_text(json.dumps(metadata))
    cfg={'metadata':str(path),'output':str(tmp_path/'hpo'),'run_name':'pod_gru','seed':42,**SETTINGS,'trials':3,
         'parallel':2,'compressor':{'name':'pod','batch_size':8},'dataset':{'Nx':2,'Ni':1,'horizon':2},
         'forecaster':{'name':'gru','hiddens':[4],'epochs':2,'patience':2,'optimizer':'adam','weight_decay':0.0,
                       'dropout':0.0,'rollout_weight':0.5,'normalization':None}}
    best,results=hpo(cfg,[0,1,2])
    assert set(results)=={0,1,2} and all(np.isfinite(r['sine']['mean_nrmse']) for r in results.values())
    assert 1<=best['compressor']['rank']<=8 and 'lr' in best['forecaster']
    rows=[json.loads(line) for line in (tmp_path/'hpo'/'pod_gru'/'hpo'/'metrics.jsonl').read_text().splitlines()]
    trials=[row for row in rows if row.get('table')=='HPO/stage2/trials'][0]['rows']
    assert sum(row[1]=='complete' for row in trials)==3
    # POD comes from seed 0 in every seed.
    bases=[pickle.loads((tmp_path/'hpo'/'pod_gru'/f'seed_{seed}'/'preprocessing.pkl').read_bytes())[1].U_r for seed in (0,1,2)]
    assert all(np.array_equal(bases[0],basis) for basis in bases[1:])
