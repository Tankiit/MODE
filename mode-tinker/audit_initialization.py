"""Compare weights-only restoration routes without optimizer updates."""
import argparse
import json
from pathlib import Path

import numpy as np
import tinker

from fresh_trajectories import load_run_data, per_example_eval
from train_sieve_tinker import atomic_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--seed',type=int,default=44)
    args=p.parse_args()
    directory=Path(args.run_dir)
    manifest=json.loads((directory/'manifest.json').read_text())
    from types import SimpleNamespace
    cfg=SimpleNamespace(**manifest['config'])
    _,_,dev,_=load_run_data(cfg,args.seed)
    initial=json.loads((directory/f'initialization-seed{args.seed}.json').read_text())
    output=directory/'initialization-audit'
    output.mkdir(exist_ok=True)
    for route in ('weights_only','explicit_seed_then_load'):
        service=tinker.ServiceClient()
        try:
            if route=='weights_only':
                client=service.create_training_client_from_state(initial['state_path'])
            else:
                client=service.create_lora_training_client(base_model=cfg.base_model,rank=cfg.rank,
                    seed=args.seed,train_mlp=True,train_attn=True,train_unembed=False)
                client.load_state(initial['state_path']).result()
            info=client.get_info().model_dump(mode='json')
            result=per_example_eval(client,dev,'validation-development',cfg.seq_len,
                                    cfg.reward_batches*cfg.batch_size*cfg.seq_len)
            differences=np.array([a['nll']-b['nll'] for a,b in zip(result['examples'],initial['development']['examples'],strict=True)])
            summary={'route':route,'seed':args.seed,'state_path':initial['state_path'],
                'optimizer_steps':0,'info':info,'nll':result['nll'],
                'mean_nll_difference':float(differences.mean()),'max_example_difference':float(abs(differences).max())}
            atomic_json(output/f'{route}-seed{args.seed}.json',{**summary,'evaluation':result})
            print(json.dumps(summary),flush=True)
            service.close('success').result()
        except Exception:
            service.close('errored').result()
            raise


if __name__=='__main__':
    main()
