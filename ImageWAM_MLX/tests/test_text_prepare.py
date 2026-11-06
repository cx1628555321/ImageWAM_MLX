"""CPU-only tests of the actual cache/controller methods; no weight load."""
import ast,importlib.util,time,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('text_cache',ROOT/'imagewam_mlx/text_cache.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class MX:
    array=staticmethod(np.array)
    eval=staticmethod(lambda x:None)
    synchronize=staticmethod(lambda:None)
# Execute the original methods without importing the GPU runtime/large weights.
tree=ast.parse((ROOT/'imagewam_mlx/runtime.py').read_text())
cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ImageWAM')
names={'clear_text_cache','text_cache_stats','_encode_text','prepare_text'}
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
ns={'mx':MX,'time':time};exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(ROOT/'imagewam_mlx/runtime.py'),'exec'),ns)
Controller=type('Controller',(),{n:ns[n] for n in names})
class Tests(unittest.TestCase):
    def make(self):
        m=Controller();m.text_cache_size=1;m._text_cache_prompt=None;calls=[]
        def encoder(ids):calls.append(ids.copy());return ids.astype(np.float32)*2
        m.qwen=mod.TextCache(encoder,capacity=1)
        m.tokenize=lambda s:np.array([[ord(c) for c in s]],dtype=np.int32)
        return m,calls
    def test_first_same_change_return(self):
        m,calls=self.make()
        for text,count in [('A',1),('A',1),('long text',2),('long text',2),('A',3)]:
            info=m.prepare_text(text);self.assertEqual(len(calls),count)
            out=m._encode_text(m.tokenize(text),text)
            np.testing.assert_array_equal(out,m.tokenize(text)*2)
            self.assertEqual(len(calls),count);self.assertEqual(len(m.qwen.entries),1)
            self.assertGreaterEqual(info['prepare_ms'],0)
    def test_raw_text_change_even_same_tokens(self):
        m,calls=self.make();m.tokenize=lambda _:np.array([[1]],np.int32)
        m.prepare_text('a');m.prepare_text('b');self.assertEqual(len(calls),2)
    def test_clear_and_disabled(self):
        m,calls=self.make();m.prepare_text('a');m.clear_text_cache();m.prepare_text('a')
        self.assertEqual(len(calls),2);m.text_cache_size=0
        with self.assertRaises(ValueError):m.prepare_text('a')
    def test_oversize_does_not_reuse_old_instruction(self):
        m,calls=self.make();m.qwen.max_bytes=8;m.prepare_text('a');m.prepare_text('too long')
        self.assertEqual(len(m.qwen.entries),0);m.prepare_text('too long');self.assertEqual(len(calls),3)
    def test_invalid_prompt_does_not_poison_cache(self):
        m,calls=self.make();m.prepare_text('a')
        def invalid(_):raise ValueError('length')
        m.tokenize=invalid
        with self.assertRaises(ValueError):m.prepare_text('bad')
        self.assertEqual(m._text_cache_prompt,'a');self.assertEqual(len(calls),1)
if __name__=='__main__':unittest.main()
