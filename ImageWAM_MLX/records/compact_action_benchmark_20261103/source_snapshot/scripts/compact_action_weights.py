"""Create an immutable action-only checkpoint by copying exact BF16 byte ranges."""
import json, struct, hashlib, shutil
from pathlib import Path
src=Path('weights/bf16'); dst=src/'action_only'
dst.mkdir(exist_ok=False)
original=(src/'manifest.json').read_bytes()
manifest=json.loads(original)
manifest['layout']='action_only_v1'
manifest['source_manifest_sha256']=hashlib.sha256(original).hexdigest()
def rewrite(name):
    with (src/name).open('rb') as f:
        size=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(size)); start=8+size
        new={}; ranges=[]; offset=0
        for key,item in header.items():
            if key=='__metadata__':continue
            shape=list(item['shape']); lo,hi=item['data_offsets']
            if name=='visual.safetensors':
                if key.startswith('single_blocks.19.linear2.') or key=='single_blocks.19.norm.query_norm.scale':continue
                if key.startswith('single_blocks.19.linear1.'):
                    assert shape[0]==27648 and item['dtype']=='BF16'
                    row=(hi-lo)//shape[0]; lo+=3072*row; hi=lo+6144*row; shape[0]=6144
                    key=key.replace('.linear1.','.kv.')
            if name=='vae.safetensors' and key.startswith('encoder.quant_conv.'):
                assert shape[0]==64 and item['dtype']=='BF16'
                hi=lo+(hi-lo)//2; shape[0]=32
            length=hi-lo
            new[key]={'dtype':item['dtype'],'shape':shape,'data_offsets':[offset,offset+length]}
            ranges.append((start+lo,length));offset+=length
        encoded=json.dumps(new,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%8)
        with (dst/name).open('xb') as g:
            g.write(struct.pack('<Q',len(encoded)));g.write(encoded)
            for pos,length in ranges:
                f.seek(pos)
                while length:
                    block=f.read(min(length,8*1024*1024))
                    if not block:raise EOFError(name)
                    g.write(block);length-=len(block)
    digest=hashlib.sha256()
    with (dst/name).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):digest.update(block)
    manifest['files'][name]={'sha256':digest.hexdigest(),'bytes':(dst/name).stat().st_size,'tensors':{k:v['shape'] for k,v in new.items()}}
for name in manifest['files']:
    if name in ('visual.safetensors','vae.safetensors'):rewrite(name)
    else:(dst/name).symlink_to('../'+name)
# Publish manifest last: incomplete conversions are never selected by the loader.
(dst/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('COMPLETE',dst,flush=True)
