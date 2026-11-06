from pathlib import Path
import shutil
p=Path('imagewam_mlx_work/project');exp=p/'experiments/remaining_20261105'
shutil.copyfile(exp/'text_cache.py',p/'imagewam_mlx/text_cache.py')
f=p/'imagewam_mlx/runtime.py';s=f.read_text();assert 'tail_visual=None):' in s
s=s.replace('tail_visual=None):','tail_visual=None,text_cache_size=0):')
needle="        mode=optimization or os.environ.get('IMAGEWAM_MLX_OPTIMIZATION','default')"
s=s.replace(needle,"        if not isinstance(text_cache_size,int) or text_cache_size<0:raise ValueError('text_cache_size must be a nonnegative integer')\n        if text_cache_size and profile!='speed':raise ValueError('Text cache currently validated only for speed profile')\n        self.text_cache_size=text_cache_size\n"+needle)
needle='    def warmup(self,samples,repeats=1):'
s=s.replace(needle,"        if text_cache_size:\n            from .text_cache import TextCache\n            self.qwen=TextCache(self.qwen,capacity=text_cache_size)\n\n    def clear_text_cache(self):\n        if self.text_cache_size:self.qwen.clear()\n\n    def text_cache_stats(self):\n        if not self.text_cache_size:return {'enabled':False}\n        return dict(enabled=True,capacity=self.qwen.capacity,max_feature_bytes=self.qwen.max_bytes,hits=self.qwen.hits,misses=self.qwen.misses,feature_bytes=self.qwen.bytes)\n\n"+needle)
f.write_text(s)
for name in ('infer.py','benchmark.py'):
 f=p/'scripts'/name;s=f.read_text();needle="p.add_argument('--no-tail-visual'";i=s.index(needle)
 extra="p.add_argument('--text-cache-size',type=int,default=0,help='Opt-in per-model text feature LRU entries; 0 disables; clear after changing weights')\n"
 if name=='benchmark.py':extra+="p.add_argument('--steps',type=int,default=10,help='Denoising steps; fewer steps change action outputs')\n"
 s=s[:i]+extra+s[i:];s=s.replace('tail_visual=False if a.no_tail_visual else None,','tail_visual=False if a.no_tail_visual else None,text_cache_size=a.text_cache_size,')
 if name=='benchmark.py':
  s=s.replace('a.iterations<1 or a.warmup<0 or a.profile_iterations<1','a.iterations<1 or a.warmup<0 or a.profile_iterations<1 or a.steps<1 or a.text_cache_size<0')
  s=s.replace('model.predict(images,state,prompt,noise,profile=', 'model.predict(images,state,prompt,noise,steps=a.steps,profile=')
  s=s.replace('model.predict(images,state,prompt,noise,trace=', 'model.predict(images,state,prompt,noise,steps=a.steps,trace=')
  s=s.replace("report['status']='completed';", "report['steps']=a.steps;report['text_cache']=model.text_cache_stats();report['status']='completed';")
 else:
  s=s.replace("if a.warmup<0:p.error('warmup must be nonnegative')", "if a.warmup<0 or a.steps<1 or a.text_cache_size<0:p.error('warmup/cache-size must be nonnegative and steps positive')")
  s=s.replace("{'staged_visual':model.staged_visual", "{'text_cache':model.text_cache_stats(),'staged_visual':model.staged_visual")
 f.write_text(s)
