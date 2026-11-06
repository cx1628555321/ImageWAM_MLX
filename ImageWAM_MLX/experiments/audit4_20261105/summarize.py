import json,collections,statistics
from pathlib import Path
r=Path('records/audit4_20261105');data=[json.loads(x) for x in (r/'gpu_buffers.jsonl').read_text().splitlines()];host=json.loads((r/'probe_host.json').read_text());rows=[]
for request in range(30):
 ds=[x for x in data if x['request']==request];valid=sorted((x['gpu_start'],x['gpu_end']) for x in ds if x['gpu_end']>x['gpu_start']>0);merged=[]
 for s,e in valid:
  if merged and s<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],e)
  else:merged.append([s,e])
 busy=sum(e-s for s,e in merged)*1000;span=(merged[-1][1]-merged[0][0])*1000 if merged else 0
 rows.append(dict(request=request,buffers=len(ds),positive_intervals=len(valid),gpu_interval_union_ms=busy,gpu_span_ms=span,between_buffer_gap_ms=span-busy,host_ms=host['instrumented_ms'][request],status_counts=dict(collections.Counter(x['status'] for x in ds))))
result=dict(gpu_rows=rows,gpu_means={k:statistics.mean(x[k] for x in rows) for k in ('buffers','positive_intervals','gpu_interval_union_ms','gpu_span_ms','between_buffer_gap_ms','host_ms')},host_uninstrumented_mean=statistics.mean(host['baseline_ms']),zero_start=sum(x['gpu_start']==0 for x in data),equal_start_end=sum(x['gpu_start']==x['gpu_end'] for x in data))
for mode in ('baseline','prewarmed','shapeless_step','shapeless_prefill'):
 p=r/mode/'results.json'
 if p.exists():
  d=json.loads(p.read_text());result[mode]=dict(errors=d['errors'],counts=d['counts'],requests=len(d['rows']),first_four=[dict(tokens=x['tokens'],ms=x['ms'],trace_delta=x['trace_delta']) for x in d['rows'][:4]],repeat_mean_ms=statistics.mean(x['ms'] for x in d['rows'][4:]) if len(d['rows'])>4 else None,all_exact=all(x.get('exact',True) for x in d['rows']) if d['rows'] else None,prewarm=d['prewarm'])
p=r/'boundary.json'
if p.exists():
 d=json.loads(p.read_text());result['boundary']=dict(wall_mean_ms=statistics.mean(x['wall_ms'] for x in d['requests']),caller_cpu_mean_ms=statistics.mean(x['caller_cpu_ms'] for x in d['requests']),events={k:dict(calls_per_request=sum(x['kind']==k for x in d['events'])/30,wall_per_request_ms=sum(x['wall_ms'] for x in d['events'] if x['kind']==k)/30,caller_cpu_per_request_ms=sum(x['caller_cpu_ms'] for x in d['events'] if x['kind']==k)/30) for k in sorted({x['kind'] for x in d['events']})})
result['export']=json.loads((r/'export/results.json').read_text()) if (r/'export/results.json').exists() else None
(r/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='gpu_rows'},indent=2))
