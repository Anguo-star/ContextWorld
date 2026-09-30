"""Complete conditional prediction error, including common prediction bias.

All conditions share the current observation and future action sequence.
Numerators and denominator are accumulated before division. Never compare raw
latent error magnitudes from different encoders as physical-state error.
"""
from __future__ import annotations
import numpy as np

ENERGIES=('baseline_energy','complete_energy','response_energy','bias_energy','wrong_energy')
RATIOS={'complete_error':'complete_energy','response_error':'response_energy',
        'common_bias_error':'bias_energy','wrong_history_error':'wrong_energy'}


def array_energies(prediction, target):
    """One scene: [condition, candidate..., latent dimension]."""
    p,z=(np.asarray(x,np.float64) for x in (prediction,target))
    if p.shape!=z.shape or p.ndim<2 or p.shape[0]<2:
        raise ValueError('Expected matching [condition>=2, ..., latent] arrays')
    if not np.isfinite(p).all() or not np.isfinite(z).all():
        raise ValueError('Non-finite predictions or targets')
    k=len(p)
    p,z=p.reshape(k,-1,p.shape[-1]),z.reshape(k,-1,z.shape[-1])
    centered=z-z.mean(axis=0,keepdims=True)
    error=p-z
    mean_error=error.mean(axis=0,keepdims=True)
    baseline=float(np.square(centered).sum())
    complete=float(np.square(error).sum())
    response=float(np.square(error-mean_error).sum())
    bias=float(k*np.square(mean_error).sum())
    wrong=float(sum(np.square(p[j]-z[i]).sum() for i in range(k) for j in range(k) if i!=j)/(k-1))
    dot=float(((p-p.mean(axis=0,keepdims=True))*centered).sum())
    assert np.isclose(complete,response+bias,rtol=1e-10,atol=1e-10)
    assert np.isclose(wrong-complete,2*k/(k-1)*dot,rtol=1e-9,atol=1e-8)
    return dict(baseline_energy=baseline,complete_energy=complete,response_energy=response,
                bias_energy=bias,wrong_energy=wrong,
                conditions=k,comparisons=p.shape[1],
                zero_separation_comparisons=int(np.count_nonzero(np.square(centered).sum(axis=(0,2))==0)))


def pair_record_energies(row):
    """Recover binary-pair geometry from existing per-condition MSE records."""
    response=row['latent_response']
    # MSE rather than summed latent error: all terms share that dimension factor.
    baseline=float(response['target_response_mse'])/2
    complete=sum(float(row[side]['correct_future_mse']) for side in ('low_strength','high_strength'))
    wrong=sum(float(row[side]['other_future_mse']) for side in ('low_strength','high_strength'))
    residual=baseline*float(response['normalized_response_error'])
    bias=complete-residual
    if baseline<=0 or min(complete,wrong)<0 or bias < -1e-8*max(1.,complete,residual):
        raise ValueError('Invalid/inconsistent paired error records')
    # Stored MSE uses float32; response geometry is accumulated in float64.
    # Near-zero differences suffer cancellation, so tolerance scales with the
    # original losses rather than with their difference.
    expected=4*baseline*response['response_gain']
    tolerance=8*np.finfo(np.float32).eps*max(wrong+complete,abs(expected),np.finfo(np.float32).tiny)
    if abs(wrong-complete-expected)>tolerance:
        raise ValueError('Cross-history error disagrees with paired response geometry')
    return dict(baseline_energy=baseline,complete_energy=complete,response_energy=residual,
                bias_energy=max(0.,bias),wrong_energy=wrong,conditions=2,comparisons=1,
                zero_separation_comparisons=0)


def aggregate(records):
    sums={k:float(sum(r[k] for r in records)) for k in ENERGIES}
    if not records or sums['baseline_energy']<=0:
        raise ValueError('Complete prediction ratio is undefined: no target separation')
    result={k:sums[v]/sums['baseline_energy'] for k,v in RATIOS.items()}
    result['history_error_reduction']=(sums['wrong_energy']-sums['complete_energy'])/sums['baseline_energy']
    assert np.isclose(result['complete_error'],result['response_error']+result['common_bias_error'],rtol=1e-9,atol=1e-9)
    return result


def bootstrap_ratios(records,indices):
    values=np.array([[r[k] for k in ENERGIES] for r in records],np.float64)
    summed=values[indices].sum(axis=1)
    if np.any(summed[:,0]<=0):
        raise ValueError('Undefined bootstrap ratio; report zero-separation coverage')
    out={k:summed[:,ENERGIES.index(v)]/summed[:,0] for k,v in RATIOS.items()}
    out['history_error_reduction']=(summed[:,4]-summed[:,1])/summed[:,0]
    return out


def summarize(records):
    result=aggregate(records)
    indices=np.random.default_rng(20260929).integers(0,len(records),(10000,len(records)))
    samples=bootstrap_ratios(records,indices)
    result['ci95']={k:np.quantile(v,[.025,.975]).tolist() for k,v in samples.items()}
    result.update(scenes=len(records),comparisons=sum(r['comparisons'] for r in records),
                  zero_separation_comparisons=sum(r['zero_separation_comparisons'] for r in records))
    return result


def paired_contrast(before,after):
    if [r['query_id'] for r in before]!=[r['query_id'] for r in after]:
        raise ValueError('Contrast scenes must align in the same order')
    indices=np.random.default_rng(20260929).integers(0,len(before),(10000,len(before)))
    samples=[bootstrap_ratios(rr,indices) for rr in (before,after)]
    estimates=[aggregate(rr) for rr in (before,after)]
    return {k:dict(after_minus_before=estimates[1][k]-estimates[0][k],
                   ci95=np.quantile(samples[1][k]-samples[0][k],[.025,.975]).tolist()) for k in estimates[0]}


def energy_growth(before,after):
    """Same encoder/candidates: separate numerator growth from denominator growth."""
    if [r['query_id'] for r in before]!=[r['query_id'] for r in after]:
        raise ValueError('Growth comparison scenes must align')
    indices=np.random.default_rng(20260929).integers(0,len(before),(10000,len(before)))
    result={}
    for key in ('complete_energy','baseline_energy'):
        x,y=(np.array([r[key] for r in rr],np.float64) for rr in (before,after))
        denominator=x[indices].sum(axis=1)
        if np.any(denominator<=0):raise ValueError('Undefined energy growth ratio')
        result[key]=dict(after_over_before=float(y.sum()/x.sum()),
            ci95=np.quantile(y[indices].sum(axis=1)/denominator,[.025,.975]).tolist())
    return result
