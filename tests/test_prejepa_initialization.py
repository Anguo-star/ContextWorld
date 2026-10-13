import copy
import pytest
import torch
from torch import nn
from stable_worldmodel.wm.prejepa.prejepa import PreJEPA
from stable_worldmodel.wm.prejepa.module import Embedder, CausalPredictor
from contextworld.training.initialization import apply_initialization, validate_initialization
from contextworld.training.prejepa_initialization import adapt_prejepa_state


def model(order):
    return PreJEPA(nn.Linear(2, 2), CausalPredictor(num_patches=1,num_frames=3,
        dim=4+2*len(order),depth=1,heads=2,dim_head=3,mlp_dim=12),
        nn.ModuleDict({k:Embedder(in_chans=2,emb_dim=2) for k in order}))

@pytest.mark.parametrize('order',[['proprio','action'],['action','observation']])
def test_project_load_and_backward_export_reload(order,tmp_path):
    source=model(order);target=model(['action']);path=tmp_path/'original.pt'
    torch.save(source.state_dict(),path)
    receipt=apply_initialization(target,validate_initialization(path))
    a=receipt['adaptation'];assert not a['function_equivalent_to_source']
    indices=[0,1,2,3]+([6,7] if order[0]=='proprio' else [4,5])
    assert a['retained_channels']==indices
    assert torch.equal(target.predictor.pos_embedding,source.predictor.pos_embedding[:,:,indices])
    assert torch.equal(target.extra_encoders['action'].patch_embed.weight,source.extra_encoders['action'].patch_embed.weight)
    opt=torch.optim.AdamW(target.parameters());assert not opt.state
    loss=target.predict(torch.randn(2,3,1,6)).square().mean();loss.backward();opt.step()
    assert torch.isfinite(loss)
    path2=tmp_path/'weights.pt';torch.save(target.state_dict(),path2)
    restored=model(['action']);r=apply_initialization(restored,validate_initialization(path2));assert r['adaptation'] is None
    for k,v in target.state_dict().items():assert torch.equal(v,restored.state_dict()[k])


def test_incompatible_state_rejected_before_mutation(tmp_path):
    source=model(['proprio','action']);target=model(['action']);before=copy.deepcopy(target.state_dict());state=source.state_dict()
    state['predictor.transformer.layers.0.0.to_qkv.weight']=torch.randn(19,8)
    p=tmp_path/'bad.pt';torch.save(state,p)
    with pytest.raises(ValueError):apply_initialization(target,validate_initialization(p))
    for k,v in before.items():assert torch.equal(v,target.state_dict()[k])
