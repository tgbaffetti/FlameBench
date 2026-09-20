import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from DataProcessing.prepare import convert
from DataProcessing.Dataset import TrainingDataset, TestDataset
from DataProcessing.scaling import FeatureScaler
from DataProcessing.latent import LatentDataset
from Baselines.OrderReduction.Linear.POD import POD
from Baselines.Forecast.Classical.ARX import ARX
from Baselines.Forecast.DL.networks import GRU, LSTM, Transformer
from Baselines.OrderReduction.DL.AE import AE
from Baselines.OrderReduction.DL.VAE import VAE
from Experiments.metrics import FieldMetrics, gain_phase, relative_l2
from Experiments.evaluation import evaluate
from Experiments.run import fit, test as run_test, hpo
from Experiments.paths import run_directory


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
        np.save(x_path,data.astype(np.float32));np.save(u_path,phi.astype(np.float32))
        cases.append({'name':name,'split':split,'waveform':'step','duration':0.04,
                      'data':str(x_path),'phi':str(u_path)})
    volumes = tmp_path/'volumes.npy'; np.save(volumes,[1.,2.,3.,4.])
    return {'dt':0.0005,'fields':['T','mix:Q'],'initial_snapshot_is_steady':True,
            'cell_volumes':str(volumes),'cases':cases}


def test_stream_conversion(tmp_path):
    raw=np.arange(5*2*13,dtype=np.float64).reshape(5,2,13)
    source,target=tmp_path/'raw.npz',tmp_path/'prepared.npy'
    np.savez_compressed(source,data=raw)
    convert(source,target,2,block_cells=2)
    result=np.load(target,mmap_mode='r')
    assert isinstance(result,np.memmap)
    np.testing.assert_array_equal(result,raw.transpose(2,1,0))


def test_no_split_or_trajectory_leakage(metadata):
    train=TrainingDataset(metadata,Nx=2,Ni=0,validation_fraction=0.25)
    val=TrainingDataset(metadata,Nx=2,Ni=0,validation_fraction=0.25,partition='validation')
    for (_,lo,hi),(_,v_lo,v_hi) in zip(train.segments(),val.segments()):
        assert hi==v_lo
    assert len(train)==2*(60-3)
    for index in [0,len(train)//2-1,len(train)//2,len(train)-1]:
        assert train[index]['history'].shape==(3,2,4)
    scaler=FeatureScaler().fit(train.snapshot_batches(7))
    full=np.concatenate(list(train.snapshot_batches(7)))
    np.testing.assert_allclose(scaler.mean,full.astype('float64').mean(axis=(0,2)),rtol=1e-6)
    # Changing validation values must not change fitted statistics.
    for case,lo,hi in val.segments():
        a=np.load(case['data'],mmap_mode='r+'); a[lo:hi]=1e6;a.flush()
    other=FeatureScaler().fit(train.snapshot_batches(5))
    np.testing.assert_allclose(other.mean,scaler.mean)
    assert len(list(DataLoader(train,batch_size=8,num_workers=0)))>0


def test_pod_arx_and_recursive_metrics(metadata,tmp_path):
    train=TrainingDataset(metadata,Nx=2)
    scaler=FeatureScaler().fit(train.snapshot_batches(8))
    pod=POD(rank=2,batch_size=8).fit(train,scaler)
    latent=LatentDataset(train,pod,scaler,tmp_path/'latent',batch_size=8)
    model=ARX(alpha=1e-8).fit(DataLoader(latent,batch_size=8))
    results=evaluate(model,TestDataset(metadata,Nx=2),pod,scaler,tmp_path/'eval')
    case=results['sine']
    assert case['forecast_steps']==80
    assert case['mean_nrmse']<1e-3
    assert case['heat_release_relative_l2']<1e-3
    q=np.load(tmp_path/'eval'/'sine_Q.npz')
    x=np.load(metadata['cases'][-1]['data'])
    np.testing.assert_allclose(q['reference'],x[1:,1]@np.array([1,2,3,4]),rtol=1e-6)


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


@pytest.mark.parametrize('cls',[GRU,LSTM,Transformer])
@pytest.mark.parametrize('Ni',[0,5])
@pytest.mark.parametrize('Nx',[0,2])
def test_neural_fit_resume(cls,Nx,Ni,tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(1)
    history=torch.randn(16,Nx+1,2)
    forcing=torch.ones(16,Ni+2)
    target=history[:,-1]*0.9
    data=DataLoader(torch.utils.data.TensorDataset(history,forcing,target),batch_size=8)
    model=cls(rank=2,Nx=Nx,Ni=Ni,hidden=8,layers=1,epochs=2)
    model.fit(data,data,directory=tmp_path)
    output=model.predict(history[:2].numpy(),forcing[:2].numpy())
    assert output.shape==(2,2) and np.isfinite(output).all()
    resumed=cls(rank=2,Nx=Nx,Ni=Ni,hidden=8,layers=1,epochs=3)
    resumed.fit(data,data,directory=tmp_path,resume=True)
    assert torch.load(tmp_path/'last.pt',weights_only=False)['epoch']==2


@pytest.mark.parametrize('cls',[AE,VAE])
def test_learned_compressor(cls,metadata):
    train=TrainingDataset(metadata,Nx=2)
    val=TrainingDataset(metadata,Nx=2,partition='validation')
    scaler=FeatureScaler().fit(train.snapshot_batches())
    compressor=cls(rank=2,hidden=8,epochs=1,batch_size=16).fit(train,scaler,validation=val)
    raw=next(train.snapshot_batches(2))
    result=compressor.decode(compressor.encode(scaler.transform(raw)))
    assert result.shape==raw.shape and np.isfinite(result).all()


def test_full_pipeline_and_hpo(metadata,tmp_path):
    path=tmp_path/'metadata.json';path.write_text(json.dumps(metadata))
    cfg={'metadata':str(path),'output':str(tmp_path/'run'),'run_name':'pod_arx_test','seed':42,'Nx':2,'Ni':0,'validation_fraction':0.2,
         'compressor':{'name':'pod','rank':2,'batch_size':8},'model':{'name':'arx','alpha':1e-6},
         'logging':{'wandb':{'mode':'disabled'}},'evaluation':{'heat_release':True}}
    score=fit(cfg)
    assert np.isfinite(score)
    assert run_test(cfg)['sine']['mean_nrmse']<0.01
    assert list((run_directory(cfg)/'tensorboard').glob('events.*'))
    with pytest.raises(FileExistsError): fit(cfg)
    cfg['output']=str(tmp_path/'hpo')
    hpo(cfg,1)
    assert (run_directory(cfg)/'best.json').exists()


def test_physical_metadata_required(metadata,tmp_path):
    metadata['initial_snapshot_is_steady']=None
    with pytest.raises(ValueError,match='initial_snapshot'):
        evaluate(None,TestDataset(metadata,Nx=2),None,None,tmp_path)
    metadata['initial_snapshot_is_steady']=True
    metadata['cell_volumes']=None
    with pytest.raises(ValueError,match='physical cell volumes'):
        evaluate(None,TestDataset(metadata,Nx=2),None,None,tmp_path)


def test_reproducible_resume(tmp_path):
    # A resumed shuffled neural fit must match the uninterrupted training path.
    from utils import seed_everything
    torch.set_num_threads(1)
    history=torch.arange(72,dtype=torch.float32).reshape(12,3,2)/72
    data=torch.utils.data.TensorDataset(history,torch.ones(12,2),history[:,-1]*0.9)
    def loaders():
        return DataLoader(data,batch_size=4,shuffle=True),DataLoader(data,batch_size=4)
    seed_everything(7)
    full=GRU(rank=2,Nx=2,Ni=0,hidden=8,layers=1,epochs=3)
    full.fit(*loaders())
    seed_everything(7)
    part=GRU(rank=2,Nx=2,Ni=0,hidden=8,layers=1,epochs=1)
    part.fit(*loaders(),directory=tmp_path)
    continued=GRU(rank=2,Nx=2,Ni=0,hidden=8,layers=1,epochs=3)
    continued.fit(*loaders(),directory=tmp_path,resume=True)
    for key,value in full.network.state_dict().items():
        torch.testing.assert_close(value,continued.network.state_dict()[key],rtol=0,atol=0)


def test_dataloader_workers(metadata):
    dataset=TrainingDataset(metadata,Nx=2,Ni=5)
    batch=next(iter(DataLoader(dataset,batch_size=4,num_workers=2)))
    assert batch['history'].shape==(4,3,2,4)
    assert batch['forcing'].shape==(4,7)
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
    assert headings==['## 1. Data preparation','## 2. Compressor','## 3. Forecast','## 4. Evaluation']
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
                             Nx=2,Ni=4,rank=2,batch_size=8,
                             logging_config={'logging':{'wandb':{'mode':'disabled'}}})
            first=False
    result=namespace['results']['sine']
    # This is a notebook execution check, not a claim that longer history improves accuracy.
    assert np.isfinite(result['mean_nrmse'])
    assert result['heat_release_status']=='explicitly_disabled'
    assert result['first_predicted_index']==1
    assert (namespace['output']/'model.pkl').exists()


@pytest.mark.parametrize('Nx,Ni',[(0,0),(4,1),(1,5),(0,4)])
def test_independent_history_alignment(metadata,tmp_path,Nx,Ni):
    for partition in ['train','validation']:
        data=TrainingDataset(metadata,Nx=Nx,Ni=Ni,partition=partition)
        scaler=FeatureScaler().fit(data.snapshot_batches(8))
        pod=POD(rank=2,batch_size=8).fit(data,scaler)
        latent=LatentDataset(data,pod,scaler,tmp_path/partition)
        assert len(latent)==len(data)
        offset=0
        for case,lo,hi in data.segments():
            fields,phi=data.arrays(case)
            context=max(Nx,Ni)+1
            for index in [offset,offset+hi-lo-context-1]:
                k=lo+context+index-offset
                sample=data[index]
                np.testing.assert_array_equal(sample['history'],fields[k-Nx-1:k])
                np.testing.assert_array_equal(sample['forcing'],phi[k-Ni-1:k+1])
                np.testing.assert_array_equal(sample['target'],fields[k])
                z,u,target=latent[index]
                np.testing.assert_allclose(z,pod.encode(scaler.transform(sample['history'].numpy())),atol=1e-5)
                np.testing.assert_array_equal(u,sample['forcing'])
                assert len(z)==Nx+1 and len(u)==Ni+2
                assert k-Nx-1>=lo and k-Ni-1>=lo
            offset+=hi-lo-context


def test_step_forcing_padding_in_rollout(metadata,tmp_path):
    from Experiments.run import validation_rollout
    from Experiments.evaluation import forcing_window
    np.testing.assert_array_equal(forcing_window(np.array([1.3,1.3,1.3]),1,4),
                                  np.array([[1,1,1,1,1.3,1.3]],dtype=np.float32))
    train=TrainingDataset(metadata,Nx=0,Ni=4)
    scaler=FeatureScaler().fit(train.snapshot_batches())
    pod=POD(rank=2,batch_size=8).fit(train,scaler)
    class Recorder:
        def __init__(self):
            self.inputs=[]
        def predict(self,history,forcing):
            self.inputs.append((history.copy(),forcing.copy()))
            return history[:,-1]
    recorder=Recorder()
    test_data=TestDataset(metadata,Nx=0,Ni=4)
    evaluate(recorder,test_data,pod,scaler,tmp_path/'eval',heat_release=False)
    phi=np.load(metadata['cases'][-1]['phi'])
    for k,(history,u) in enumerate(recorder.inputs[1:],start=1):
        assert history.shape==(1,1,2)
        expected=[1 if t<0 else phi[t] for t in range(k-5,k+1)]
        np.testing.assert_array_equal(u[0],np.asarray(expected,dtype=np.float32))
    observed=Recorder()
    result=evaluate(observed,test_data,pod,scaler,tmp_path/'observed',
                    heat_release=False,initialization='observed_history')
    assert result['sine']['first_predicted_index']==5
    fields=np.load(metadata['cases'][-1]['data'])
    np.testing.assert_allclose(observed.inputs[0][0][0],pod.encode(scaler.transform(fields[4:5])),atol=1e-6)
    np.testing.assert_array_equal(observed.inputs[0][1][0],phi[:6])
    validation=TrainingDataset(metadata,Nx=0,Ni=4,partition='validation')
    latent=LatentDataset(validation,pod,scaler,tmp_path/'latent')
    recorder=Recorder()
    validation_rollout(recorder,latent)
    validation_phi=np.load(latent.paths[0][1])
    np.testing.assert_array_equal(recorder.inputs[0][1][0],validation_phi[:6])


@pytest.mark.parametrize('Nx,Ni',[(-1,0),(0,-1),(1.5,0),(0,True)])
def test_invalid_histories(metadata,Nx,Ni):
    with pytest.raises(ValueError,match='nonnegative integers'):
        TrainingDataset(metadata,Nx=Nx,Ni=Ni)


@pytest.mark.parametrize('Nx,Ni',[(0,0),(4,1),(1,5)])
def test_forecaster_inputs_alignment(Nx,Ni):
    from Baselines.Forecast.DL.networks import forecaster_inputs
    history=torch.arange(2*(Nx+1)*3,dtype=torch.float32).reshape(2,Nx+1,3)
    forcing=torch.arange(2*(Ni+2),dtype=torch.float32).reshape(2,Ni+2)+1
    inputs=forecaster_inputs(history,forcing,Nx,Ni)
    assert inputs.shape==(2,max(Nx,Ni)+2,6)
    for index,time in enumerate(range(-max(Nx,Ni),2)):
        if -Nx<=time<=0:
            torch.testing.assert_close(inputs[:,index,:3],history[:,time+Nx])
            assert (inputs[:,index,4]==1).all()
        else:
            assert (inputs[:,index,:3]==0).all()
            assert (inputs[:,index,4]==0).all()
        if time>=-Ni:
            torch.testing.assert_close(inputs[:,index,3],forcing[:,time+Ni]-1)
            assert (inputs[:,index,5]==1).all()
        else:
            assert (inputs[:,index,3]==0).all()
            assert (inputs[:,index,5]==0).all()
    # The target-time token contains only prescribed forcing, never a future state.
    assert (inputs[:,-1,:3]==0).all()
    assert (inputs[:,-1,4]==0).all()
    torch.testing.assert_close(inputs[:,-1,3],forcing[:,-1]-1)


@pytest.mark.parametrize('cls',[GRU,LSTM,Transformer])
@pytest.mark.parametrize('Nx,Ni',[(0,0),(4,1),(1,5)])
def test_forcing_enters_neural_core(cls,Nx,Ni):
    torch.manual_seed(19)
    model=cls(rank=2,Nx=Nx,Ni=Ni,hidden=8,layers=1)
    network=model.network
    network.eval()
    history=torch.randn(2,Nx+1,2)
    forcing=torch.randn(2,Ni+2,requires_grad=True)
    core_outputs=[]
    def capture_core(module,args,output):
        core_outputs.append(output if cls is Transformer else output[0])
    hook=network.core.register_forward_hook(capture_core)
    prediction=network(history,forcing)
    # Every supplied forcing value must affect the core, before the prediction head.
    core_gradient=torch.autograd.grad(core_outputs[0][:,-1,0].sum(),forcing,retain_graph=True)[0]
    assert torch.isfinite(core_gradient).all()
    assert (core_gradient.abs().sum(dim=0)>0).all()
    prediction.sum().backward()
    assert (forcing.grad.abs().sum(dim=0)>0).all()
    with torch.no_grad():
        changed=network(history,forcing.detach()+0.5)
    assert not torch.allclose(prediction,changed)
    hook.remove()


def test_multi_seed_cli_layout(metadata,tmp_path,monkeypatch):
    import sys
    from Experiments.run import main
    metadata_path=tmp_path/'metadata.json'
    metadata_path.write_text(json.dumps(metadata))
    cfg={'metadata':str(metadata_path),'output':str(tmp_path/'results'),'Nx':2,'Ni':0,
         'validation_fraction':0.2,'compressor':{'name':'pod','rank':2,'batch_size':8},
         'model':{'name':'arx','alpha':1e-6},'logging':{'wandb':{'mode':'disabled'}},
         'evaluation':{'heat_release':True}}
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
        assert any('validation/rollout_latent_mse' in row for row in records)
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
    cfg['trial']=3
    assert run_directory(cfg).name=='trial_0003'
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
