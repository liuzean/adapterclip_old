"""Optional correctness checks. Not executed automatically or during development."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'models_test'))
import torch
from context_alignment_utils import ContextMatcher, RawTap, descriptors, nearest_distance


class ContextAlignmentTests(unittest.TestCase):
    def test_kernel_one_excludes_cls_and_preserves_input(self):
        value=torch.arange(30,dtype=torch.float32).reshape(2,5,3)+1
        before=value.clone()
        expected=torch.nn.functional.normalize(value[:,1:],dim=-1)
        self.assertTrue(torch.equal(descriptors(value,1),expected))
        self.assertTrue(torch.equal(value,before))

    def test_pool_never_mixes_support_images(self):
        tokens=torch.zeros(2,5,2)
        tokens[0,:,0]=1
        tokens[1,:,1]=1
        result=descriptors(tokens,3)
        self.assertTrue(torch.allclose(result[0],torch.tensor([1.,0.]).expand(4,2)))
        self.assertTrue(torch.allclose(result[1],torch.tensor([0.,1.]).expand(4,2)))

    def test_chunked_match_equals_full_search(self):
        generator=torch.Generator().manual_seed(4)
        q=torch.nn.functional.normalize(torch.randn(7,9,generator=generator),dim=-1)
        s=torch.nn.functional.normalize(torch.randn(13,9,generator=generator),dim=-1)
        expected=(1-q@s.T).min(dim=-1).values
        for chunk in (1,3,20):
            self.assertTrue(torch.allclose(nearest_distance(q,s,chunk),expected,atol=1e-6))

    def test_invalid_grid_and_zero_descriptors_fail(self):
        with self.assertRaises(ValueError):
            descriptors(torch.ones(1,7,3),3)
        with self.assertRaises(ValueError):
            descriptors(torch.zeros(1,5,3),1)

    def test_raw_hook_does_not_replace_or_modify_output(self):
        class FakeTransformer(torch.nn.Module):
            def forward(self,x):
                return [x,x],[x]*4
        model=FakeTransformer()
        value=torch.ones(5,2,1024)
        tap=RawTap(model)
        try:
            output=model(value)
            raw=tap.take()
            self.assertIs(output[1][0],value)
            self.assertEqual(raw[0].shape,(2,5,1024))
            raw[0].zero_()
            self.assertTrue(torch.equal(value,torch.ones_like(value)))
        finally:
            tap.close()
        self.assertEqual(len(model._forward_hooks),0)

    def test_category_specific_reference_routing(self):
        def tokens(width,index,batch=1):
            result=torch.zeros(batch,5,width)
            result[:,:,index]=1
            return result
        projected={c:[tokens(768,i) for _ in range(4)] for c,i in [('a',0),('b',1)]}
        raw={c:[tokens(1024,i) for _ in range(4)] for c,i in [('a',0),('b',1)]}
        matcher=ContextMatcher(projected,raw,2)
        p=[tokens(768,0,2) for _ in range(4)]
        r=[tokens(1024,0,2) for _ in range(4)]
        old=[torch.ones(2,1,2,2) for _ in range(4)]
        out=matcher.alignments(p,r,['a','b'],old,2)
        self.assertTrue(torch.equal(out['C_1024_point'][0],torch.zeros(2,2)))
        self.assertTrue(torch.equal(out['C_1024_point'][1],torch.ones(2,2)))
        self.assertTrue(torch.equal(p[0],tokens(768,0,2)))


if __name__=='__main__':
    unittest.main()
