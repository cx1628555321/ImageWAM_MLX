"""Small MLX tests for positional integrity, budgets and opt-in lifecycle."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from types import SimpleNamespace
import numpy as np
import mlx.core as mx
from imagewam_mlx.visionzip import Config, PRESETS, REGIONS, select_tokens, install, build_prefill

class Tests(unittest.TestCase):
    def test_rank_oracle_ties_and_spatial_order(self):
        rng=np.random.default_rng(93);scores=np.round(rng.random(288),1).astype(np.float32)
        f=np.arange(288*3,dtype=np.float32).reshape(1,288,3)
        for cfg in (PRESETS['prune252'],PRESETS['prune216'],Config('identity',keep=288)):
            actual,positions=select_tokens(mx.array(f),mx.array(scores),cfg);mx.eval(actual,positions)
            expected=np.sort(np.concatenate([g[np.lexsort((g,-scores[g]))[:k]] for g,k in zip(REGIONS,cfg.quotas())]))
            np.testing.assert_array_equal(np.array(positions),expected)
            np.testing.assert_array_equal(np.array(actual),f[:,expected])

    def test_context_merge_preserves_constant_features_and_camera_budget(self):
        scores=mx.arange(288,dtype=mx.float32)
        f=mx.full((1,288,16),3.0,mx.bfloat16)
        actual,positions=select_tokens(f,scores,PRESETS['merge252']);mx.eval(actual,positions)
        ix=np.array(positions)
        self.assertEqual(actual.shape,(1,252,16))
        np.testing.assert_array_equal(np.array(actual.astype(mx.float32)),np.full((1,252,16),3.0))
        self.assertEqual([int(np.isin(ix,g).sum()) for g in REGIONS],[168,42,42])
        self.assertEqual(len(set(ix.tolist())),252)
        self.assertTrue(np.all(np.diff(ix)>0))

    def test_dynamic_budget_keeps_all_wrists(self):
        for text in range(1,160):
            cfg=PRESETS['late_align32'].resolved(text+1)
            self.assertEqual((text+1+cfg.keep)%32,0)
            self.assertTrue(257<=cfg.keep<=288)
            self.assertEqual(cfg.quotas()[1:],(48,48))

    def test_install_off_restore_no_wrapper_stacking(self):
        original=lambda *_:None
        model=SimpleNamespace(prefill=original,inference_profile='speed',prefix_precision='all_int8',
                              action=SimpleNamespace(kind='w4'),compact=True)
        install(model,'off');self.assertIs(model.prefill,original)
        install(model,'merge252');install(model,'late_align32')
        self.assertIs(model._visionzip_original_prefill,original)
        install(model,'off');self.assertIs(model.prefill,original)

    def test_reject_unsupported_modes_before_mutating(self):
        model=SimpleNamespace(prefill=lambda *_:None)
        original=model.prefill
        with self.assertRaises(ValueError):install(model,'merge252')
        with self.assertRaises(ValueError):install(model,'unknown')
        self.assertIs(model.prefill,original)
        for options in [dict(keep=95),dict(cut_after=25),dict(cut_after=1.5),dict(align32=True),dict(kv_only=True,contextual=12)]:
            with self.assertRaises(ValueError):Config('invalid',**options)

    def test_reject_nonstandard_latent_grid(self):
        fn=build_prefill(None,PRESETS['merge252'])
        with self.assertRaises(ValueError):fn(mx.zeros((1,39,7680)),mx.zeros((1,17,16,128)),mx.zeros((1,1,14)))

    def test_layout_capture_requires_cut_capture(self):
        with self.assertRaises(ValueError):
            build_prefill(None,PRESETS['merge252'],capture_layout=True)

if __name__=='__main__':unittest.main()
