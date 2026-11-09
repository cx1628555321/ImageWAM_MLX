"""Collect audited evidence into compact JSON and CSV tables, without a model."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'records/visionzip_20261109'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(name):
    p=OUT/name/'results.json';r=json.loads(p.read_text());audit=json.loads((p.parent/'independent_audit.json').read_text())
    assert r['status']=='complete' and audit['passed'] and audit['results_sha256']==sha(p)
    return r
v=read('validation');b=read('benchmark')
assert b['validation_sha256']==sha(OUT/'validation/results.json')
before=json.loads((OUT/'core_hashes_before.json').read_text())
assert all(sha(ROOT/name)==value for name,value in before.items())
out={'status':'complete','model':v['model'],'original_runtime_files_unchanged':True,
     'baseline_count':v['baseline']['count'],'identity_count':v['identity_count'],
     'observations':len(v['observations']),'cohorts':{},'precision':{},'performance':{},
     'limitations':b['limitations']+['No closed-loop task success evaluation or user-specified acceptable action-error threshold.'],
     'sources':{name:sha(OUT/name/'results.json') for name in ('validation','benchmark')}}
for cohort in ('replay','new_observations'):
    obs=[x for x in v['observations'] if x['cohort']==cohort]
    out['cohorts'][cohort]={'episodes':len({x['episode'] for x in obs}),'observations':len(obs),'calls_per_variant':sum(len(x['seeds']) for x in obs)}
labels={x['label']:x for x in v['observations']}
tables=OUT/'tables';tables.mkdir(exist_ok=True)
def write_csv(name,rows):
    with (tables/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
precision=[];e2e=[];modules=[];micro=[];worst=[]
for name,result in v['variants'].items():
    out['precision'][name]={'summary':result['summary'],'prior_project_byte_equal_count':result['prior_project_byte_equal_count'],
                            'diagnostics_by_text_length':{t:{k:x for k,x in d.items() if k!='original_positions'} for t,d in result['diagnostics_by_text_length'].items()}}
    for cohort,summary in result['summary'].items():
        precision.append({'preset':name,'cohort':cohort,'count':summary['normalized']['count'],
            'normalized_rmse':summary['normalized']['rmse'],'normalized_max_abs':summary['normalized']['worst_max_abs'],
            'raw_rmse':summary['action']['rmse'],'raw_max_abs':summary['action']['worst_max_abs'],
            'normalized_case_rmse_p95':summary['normalized']['case_rmse_p95']})
    selected=sorted(result['rows'],key=lambda x:x['normalized']['max_abs'],reverse=True)[:5]
    for row in selected:
        case=labels[row['case'].rsplit('_s',1)[0]]
        worst.append({'preset':name,'case':row['case'],'cohort':row['cohort'],'instruction':case['instruction'],
            'normalized_rmse':row['normalized']['rmse'],'normalized_max_abs':row['normalized']['max_abs'],
            'raw_max_abs':row['action']['max_abs']})
for name,records in b['e2e'].items():
    summary=records['summary'];ci=summary['block_bootstrap95_saving_ms']
    out['performance'][name]={'e2e':summary,'mean_gain_supported_in_this_run':ci[0]>0,
        'rounds':records['round_summary'],'by_observation':{c['label']:records[c['label']]['summary'] for c in b['selected']},
        'paired_modules':b['modules'][name]['summary'],'cold_text':b['cold_text'][name]['summary'],
        'micro':{label:{op:{k:x for k,x in cost.items() if k!='samples_ms'} for op,cost in rec['costs'].items()} for label,rec in b['micro'][name].items()},
        'worst_case_traces':b['worst_case_traces'][name]}
    for label,s in [('all',summary)]+[(c['label'],records[c['label']]['summary']) for c in b['selected']]:
        e2e.append({'preset':name,'observation':label,**{k:x for k,x in s.items() if k!='block_bootstrap95_saving_ms'},
            'saving_ci95_low_ms':s['block_bootstrap95_saving_ms'][0],'saving_ci95_high_ms':s['block_bootstrap95_saving_ms'][1]})
    for module,s in b['modules'][name]['summary'].items():
        modules.append({'preset':name,'module':module,**{k:x for k,x in s.items() if k!='block_bootstrap95_saving_ms'},
            'saving_ci95_low_ms':s['block_bootstrap95_saving_ms'][0],'saving_ci95_high_ms':s['block_bootstrap95_saving_ms'][1]})
    for label,rec in b['micro'][name].items():
        for op,cost in rec['costs'].items():micro.append({'preset':name,'observation':label,'tokens':rec['tokens'],'visual_keep':rec['visual_keep'],
            'operation':op,**{k:x for k,x in cost.items() if k!='samples_ms'}})
for name,rows in [('precision.csv',precision),('end_to_end.csv',e2e),('modules.csv',modules),('compression_cost.csv',micro),('worst_cases.csv',worst)]:write_csv(name,rows)
out['selected_timing_observations']=b['selected']
(OUT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'cohorts':out['cohorts'],'e2e':{k:x['e2e'] for k,x in out['performance'].items()}},indent=2))
