"""Export scientific figures from recorded values; does not run inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('records',type=Path);a=p.parse_args()
out=a.records/'figures';out.mkdir(exist_ok=True)
v=json.loads((a.records/'validation/results.json').read_text());assert v['status']=='complete'
names=['merge252','late_align32','prune252','prune216']
colors=['#2563eb','#d97706','#059669','#dc2626']
fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
x=np.arange(len(names));width=.36
for offset,cohort,title in [(-width/2,'replay','Replay: 72 calls'),(width/2,'new_observations','New observations: 288 calls')]:
    vals=[v['variants'][n]['summary'][cohort]['normalized']['rmse'] for n in names]
    axes[0].bar(x+offset,vals,width,label=title)
axes[0].set_xticks(x,names,rotation=15);axes[0].set_ylabel('Normalized action RMSE vs uncompressed')
axes[0].set_title('Output deviation by observation cohort');axes[0].legend(fontsize=8)
for name,color in zip(names,colors):
    y=v['variants'][name]['summary']['all']['normalized']['per_step_rmse']
    axes[1].plot(range(1,17),y,label=name,color=color,marker='.',linewidth=1.5)
axes[1].set_xlabel('Position in the 16-action output chunk');axes[1].set_ylabel('Normalized action RMSE')
axes[1].set_title('Output deviation by action position, 360 calls');axes[1].legend(fontsize=8)
for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
fig.suptitle('ImageWAM original-project adaptation: offline output deviation',fontsize=12)
for ext in ('svg','png'):fig.savefig(out/f'precision.{ext}',dpi=180)
plt.close(fig)
benchmark=a.records/'benchmark/results.json'
if benchmark.exists():
    b=json.loads(benchmark.read_text())
    if b['status']=='complete':
        ns=list(b['e2e']);fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
        for i,name in enumerate(ns):
            s=b['e2e'][name]['summary'];low,high=s['block_bootstrap95_saving_ms'];mean=s['mean_saving_ms']
            axes[0].errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=4,color=colors[names.index(name)])
            vals=[b['e2e'][name][c['label']]['summary']['saving_pct'] for c in b['selected']]
            axes[1].plot(range(1,len(vals)+1),vals,marker='o',label=name,color=colors[names.index(name)])
        axes[0].axvline(0,color='gray',linewidth=1);axes[0].set_yticks(range(len(ns)),ns)
        axes[0].set_xlabel('Paired mean latency saving (ms), block bootstrap 95% interval')
        axes[0].set_title('400 paired requests per configuration')
        axes[1].axhline(0,color='gray',linewidth=1);axes[1].set_xlabel('Observation index (see benchmark/results.json)')
        axes[1].set_ylabel('Mean latency reduction (%)');axes[1].set_title('Eight observations, two passes');axes[1].legend(fontsize=8)
        fig.suptitle('Warmed end-to-end inference; shared server, all outliers retained',fontsize=12)
        for ext in ('svg','png'):fig.savefig(out/f'performance.{ext}',dpi=180)
        plt.close(fig)
print(out)
