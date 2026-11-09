"""Combine audited default-allocator results into JSON and flat CSV tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    def read(path):return json.loads((a.root/path).read_text())
    assert read('suite.json')['status']=='complete'
    names=['validation','benchmark','modules_direct','zip_cost','repeatability_1','repeatability_2']
    results={}
    for name in names:
        raw=a.root/name/'results.json';r=json.loads(raw.read_text())
        audit=read(name+'/independent_audit.json')
        assert r['status']=='complete' and audit['passed']
        assert hashlib.sha256(raw.read_bytes()).hexdigest()==audit['results_sha256']
        assert r['model']['allocator_setter_guard'] and r['model']['cache_limit_bytes'] is None
        results[name]=r
    perf=results['benchmark'];valid=results['validation'];direct=results['modules_direct'];zipped=results['zip_cost']
    stability=read('stability/results.json')
    assert stability['validation_sha256']==hashlib.sha256((a.root/'validation/results.json').read_bytes()).hexdigest()
    out={'status':'complete','allocator_policy':perf['model']['allocator_policy'],
         'observations':len(valid['observations']),'calls_per_precision_variant':valid['baseline']['count'],
         'performance_observations':perf['selected'],'variants':{},'notes':perf['limitations']}
    tables={'performance':[],'modules':[],'accuracy':[],'zip':[]}
    for name,v in valid['variants'].items():
        ep=perf['e2e'][name]['summary'];accuracy=v['summary']['all']['normalized']
        data={'e2e':ep,'accuracy':accuracy,'precision_stability':stability['variants'][name],
              'profiled_request_modules':perf['modules'][name]['summary'],
              'isolated_prefix':direct['prefix'][name]['summary'],
              'isolated_action':direct['action'][name]['summary'],
              'cold_text':perf['cold_text'][name]['summary'],'zip_shapes':{},
              'repeated_calls':sum(sum(row['preset']==name for row in results[key]['rows']) for key in ('repeatability_1','repeatability_2'))}
        tables['performance'].append({'preset':name,**{k:v for k,v in ep.items() if not isinstance(v,list)},
                                      'saving_ci95_low_ms':ep['block_bootstrap95_saving_ms'][0],
                                      'saving_ci95_high_ms':ep['block_bootstrap95_saving_ms'][1]})
        tables['accuracy'].append({'preset':name,'calls':v['count'],'normalized_rmse':accuracy['rmse'],
                                   'normalized_max_abs':accuracy['worst_max_abs'],'relative_l2':accuracy['relative_l2'],
                                   'repeatability_calls':data['repeated_calls']})
        for stage,values in data['profiled_request_modules'].items():
            tables['modules'].append({'preset':name,'measurement':'profiled_request','stage':stage,
                'baseline_ms':values['baseline_mean_ms'],'candidate_ms':values['candidate_mean_ms'],
                'saving_ci95_low_ms':values['block_bootstrap95_saving_ms'][0],
                'saving_ci95_high_ms':values['block_bootstrap95_saving_ms'][1]})
        for stage in ('prefix','action'):
            values=data['isolated_'+stage]
            tables['modules'].append({'preset':name,'measurement':'isolated_module','stage':stage,
                'baseline_ms':values['baseline_mean_ms'],'candidate_ms':values['candidate_mean_ms'],
                'saving_ci95_low_ms':values['block_bootstrap95_saving_ms'][0],
                'saving_ci95_high_ms':values['block_bootstrap95_saving_ms'][1]})
        for label,rec in zipped['results'][name].items():
            row={'preset':name,'case':label,'text_tokens':rec['text_tokens'],'visual_keep':rec['visual_keep'],
                 'skip_zip':rec['skip_zip'],
                 'full_boundary_ms':rec['costs']['full_boundary']['mean_ms'],
                 'identity_dispatch_ms':rec['costs']['identity_dispatch']['mean_ms'],
                 'incremental_ms':rec['full_boundary_minus_identity_mean_ms']}
            for op in ('attention_score','selection_and_merge','layout_and_position'):
                row[op+'_ms']=rec['costs'].get(op,{}).get('mean_ms')
            tables['zip'].append(row);data['zip_shapes'][label]=row
        out['variants'][name]=data
    out['repeatability_total_calls']=sum(results[k]['calls'] for k in ('repeatability_1','repeatability_2'))
    out['all_repeated_outputs_byte_equal']=all(results[k]['all_outputs_byte_equal'] for k in ('repeatability_1','repeatability_2'))
    for name,rows in tables.items():
        with (a.root/('summary_'+name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (a.root/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print(json.dumps(tables['performance'],indent=2))


if __name__=='__main__':main()
