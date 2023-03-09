from typing import List
import pytest
import json
from tqdm import tqdm
from pathlib import Path
from itertools import product
import numpy as np
import torch as ch

from trak import TRAKer
from trak.projectors import BasicProjector

from .utils import construct_rn9, get_dataloader, eval_correlations

def get_projector(use_cuda_projector):
    if use_cuda_projector:
        return None
    return BasicProjector(grad_dim=11689512, proj_dim=4096, seed=0, proj_type='rademacher',
                          device='cuda:0')

PARAM = list(product([True, False], # serialize
                     [True, False], # basic / cuda projector
                     [32, 100], # batch size
        ))

@pytest.mark.parametrize("serialize, use_cuda_projector, batch_size", PARAM)
@pytest.mark.cuda
def test_cifar_acc(serialize, use_cuda_projector, batch_size, tmp_path):
    device = 'cuda:0'
    projector = get_projector(use_cuda_projector)
    model = construct_rn9().to(memory_format=ch.channels_last).to(device)
    model = model.eval()

    loader_train = get_dataloader(batch_size=batch_size, split='train')
    loader_val = get_dataloader(batch_size=batch_size, split='val')

    # TODO: put this on dropbox as well
    CKPT_PATH = '/mnt/xfs/projects/trak/checkpoints/resnet9_cifar2/debug'
    ckpt_files = list(Path(CKPT_PATH).rglob("*.pt"))
    ckpts = [ch.load(ckpt, map_location='cpu') for ckpt in ckpt_files]

    traker = TRAKer(model=model,
                  task='image_classification',
                  train_set_size=10_000,
                  save_dir=tmp_path,
                  device=device)

    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
        for batch in tqdm(loader_train, desc='Computing TRAK embeddings...'):
            traker.featurize(batch=batch, num_samples=len(batch[0]))

    traker.finalize_features()

    if serialize:
        del traker
        traker = TRAKer(model=model,
                    task='image_classification',
                    train_set_size=10_000,
                    save_dir=tmp_path,
                    device=device)

    for model_id, ckpt in enumerate(ckpts):
        traker.load_checkpoint(checkpoint=ckpt, model_id=model_id)
        for batch in tqdm(loader_val, desc='Scoring...'):
                traker.score(batch=batch, num_samples=len(batch[0]))

    scores = traker.finalize_scores()

    avg_corr = eval_correlations(infls=scores, tmp_path=tmp_path)
    assert avg_corr > 0.058, 'correlation with 3 CIFAR-2 models should be >= 0.058'