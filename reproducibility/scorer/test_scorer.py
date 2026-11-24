"""Synthetic specification tests, not a reproduction of the 1,039-image run."""
import math
import unittest
import bbox_scorer as s

F='D1_Artifact/A2_StructuralArtifact'
T='D2_TextCorrectness/A1_TextVisualIntegrity'

def gt(sid='g',boxes=((10,10,10,10),),families=None,width=100,height=100):
    families=families or [F]*len(boxes)
    return dict(sample_id=sid,image_width=width,image_height=height,
                annotation=dict(no_error=not bool(boxes),boxes=[dict(bbox_xywh=list(b),error_type=dict(code=f)) for b,f in zip(boxes,families)]))

def pred(boxes=(),sid='g',status='ok',families=None):
    families=families or [F]*len(boxes)
    return dict(sample_id=sid,status=status,boxes=[dict(bbox_1000_xywh=[10*v for v in b],family=f) for b,f in zip(boxes,families)])

class GeometryTests(unittest.TestCase):
    def test_union_overlap(self):
        self.assertEqual(s.union_area([(0,0,10,10),(5,0,15,10)]),150)
    def test_union_not_envelope(self):
        self.assertEqual(s.union_area([(0,0,10,10),(20,0,30,10)]),200)
    def test_tau(self):
        self.assertAlmostEqual(s.tolerance(0),1.60)
        self.assertAlmostEqual(s.tolerance(.01),1.25+.35/math.e)
    def test_exact_pair(self):
        self.assertEqual(s.pair_stats((0,0,10,10),(0,0,10,10),10000)['score'],1)
    def test_pair_area_penalty(self):
        p=s.pair_stats((0,0,20,20),(0,0,10,10),10000)
        expected=.4**.7*math.exp(-4*(math.log(4)-math.log(s.tolerance(.01)))**2)
        self.assertAlmostEqual(p['score'],expected)

class ScoringTests(unittest.TestCase):
    def test_perfect(self):
        r=s.score_sample(gt(),pred([(10,10,10,10)]))
        self.assertEqual((r['P'],r['R'],r['F']),(1,1,1))
    def test_missing_flawed(self):
        r=s.score_sample(gt(),None)
        self.assertEqual(r['F'],0);self.assertTrue(r['failed'])
    def test_clean_failure(self):
        r=s.score_sample(gt(boxes=()),pred([(0,0,10,10)],status='error'))
        self.assertEqual(r['F'],1);self.assertEqual(r['N'],0)
    def test_clean_exp(self):
        r=s.score_sample(gt(boxes=()),pred([(0,0,10,10),(20,0,10,10)]))
        self.assertAlmostEqual(r['F'],math.exp(-2))
    def test_duplicate_zero_credit(self):
        r=s.score_sample(gt(),pred([(10,10,10,10)]*2))
        self.assertEqual(r['owners'],[0]);self.assertEqual(r['credits'],[1,0])
        self.assertAlmostEqual(r['F'],2/3);self.assertEqual(r['duplicate_fp'],1)
    def test_coverage_priority(self):
        r=s.score_sample(gt(),pred([(0,0,30,30),(10,10,9,10)]))
        self.assertEqual(r['owners'],[0])
    def test_quality_breaks_equal_coverage(self):
        r=s.score_sample(gt(),pred([(0,0,30,30),(10,10,10,10)]))
        self.assertEqual(r['owners'],[1])
    def test_compact_merge(self):
        r=s.score_sample(gt(boxes=((0,0,10,10),(10,0,10,10))),pred([(0,0,20,10)]))
        self.assertEqual(r['qualities'],[1,1]);self.assertEqual(r['F'],1)
        self.assertEqual(r['merged_predictions'],1)
    def test_gapped_merge_lower_quality(self):
        r=s.score_sample(gt(boxes=((0,0,10,10),(20,0,10,10))),pred([(0,0,30,10)]))
        self.assertLess(r['F'],1);self.assertGreater(r['F'],0)
    def test_unknown_family_geometry_unchanged(self):
        r=s.score_sample(gt(),pred([(10,10,10,10)],families=['unknown']))
        self.assertEqual(r['F'],1);self.assertEqual(r['family_denominator'],0)
    def test_scale_invariance(self):
        a=s.score_sample(gt(),pred([(10,10,10,10)]))
        b=s.score_sample(gt(boxes=((20,20,20,20),),width=200,height=200),pred([(10,10,10,10)]))
        self.assertEqual(a['qualities'],b['qualities'])
    def test_bad_gt_rejected(self):
        with self.assertRaises(ValueError):s.score_sample(gt(boxes=((95,0,10,10),)),pred())
    def test_no_overlap_partition(self):
        r=s.score_sample(gt(),pred([(50,50,10,10)]))
        self.assertEqual(r['zero_overlap'],1);self.assertEqual(r['background_fp'],1)
    def test_description_does_not_score(self):
        p=pred([(10,10,10,10)]);p['boxes'][0]['description']='arbitrary text'
        self.assertEqual(s.score_sample(gt(),p)['F'],1)

class AggregateTests(unittest.TestCase):
    def test_always_empty_score_nine(self):
        a=[s.score_sample(gt(),pred()),s.score_sample(gt('c',boxes=()),pred(sid='c'))]
        self.assertAlmostEqual(s.aggregate(a)['region_bbox_score'],9)
    def test_perfect_score(self):
        a=[s.score_sample(gt(),pred([(10,10,10,10)])),s.score_sample(gt('c',boxes=()),pred(sid='c'))]
        self.assertEqual(s.aggregate(a)['region_bbox_score'],100)
    def test_family_diagnostics(self):
        a=[s.score_sample(gt(boxes=((0,0,10,10),(30,0,10,10)),families=[F,T]),pred([(0,0,10,10)])),s.score_sample(gt('c',boxes=()),pred(sid='c'))]
        r=s.aggregate(a);self.assertEqual(r['artifact_region_bbox_score'],100);self.assertEqual(r['text_region_bbox_score'],0)
    def test_bootstrap_determinism(self):
        a=[s.score_sample(gt(),pred()),s.score_sample(gt('c',boxes=()),pred(sid='c'))]
        self.assertEqual(s.bootstrap(a,20,19),s.bootstrap(a,20,19))
    def test_quality_threshold_inclusive(self):
        r=s.score_sample(gt(),pred());r['qualities']=[.1]
        self.assertEqual(s.aggregate([r])['recall_at_0_1'],1)

if __name__=='__main__':unittest.main()
