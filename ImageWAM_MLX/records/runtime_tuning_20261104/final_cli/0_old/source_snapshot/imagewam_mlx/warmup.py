"""Explicit, opt-in warmup with actual samples; no inference feature cache."""
import time
import mlx.core as mx

def warmup(model, samples, repeats=1):
    """Run dict samples accepted by predict before serving requests.

    Every sample must contain images, state, prompt and noise; steps is optional.
    Unknown future shapes continue through normal prediction without padding or
    truncation. We retain timing metadata only, not predictions or input features.
    This mutates the model's compiled-function caches: do not call concurrently
    with prediction. GPU compilation is process-local, not a saved binary cache.
    """
    if not isinstance(repeats,int) or isinstance(repeats,bool) or repeats<1:
        raise ValueError('repeats must be a positive integer')
    records=[]
    try:
        for sample in samples:
            kwargs={k:sample[k] for k in ('images','state','prompt','noise')}
            kwargs['steps']=sample.get('steps',10)
            tokens=int(model.tokenize(kwargs['prompt']).shape[1])
            durations=[]
            for _ in range(repeats):
                mx.synchronize();start=time.perf_counter()
                model.predict(**kwargs)
                mx.synchronize();durations.append((time.perf_counter()-start)*1000)
            records.append({'tokens':tokens,'horizon':int(kwargs['noise'].shape[1]),
                            'steps':kwargs['steps'],'durations_ms':durations})
    finally:
        model.trace={};model.profile={}
    return records
