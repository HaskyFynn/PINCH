import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from pinch.config import Settings
from pinch.registry import Registry, MarkerProfile
from pinch.enrollment import Enrollment, VIEWS
from pinch.identity import Matcher, IdentityManager
from pinch.vision import crop_quality


def vector(i):
    v = np.zeros(128,dtype=np.float32)
    v[i] = 1.
    return v


def registry(n=4):
    return Registry([MarkerProfile(chr(65+i),[vector(i).tolist()],.8) for i in range(n)])


class RegistryTests(unittest.TestCase):
    def test_legacy_exact_match_without_density(self):
        for density in (False,True):
            m = Matcher(registry(),use_density=density)
            self.assertEqual(m.candidate(vector(0),.01)[0],'A')

    def test_missing_enrollment_between_valid_profiles(self):
        r = registry(3)
        r.markers[0].enroll_embs=[vector(0).tolist()]
        r.markers[2].enroll_embs=[vector(2).tolist()]
        m=Matcher(Registry.from_json(r.to_json()))
        self.assertEqual(m.candidate(vector(2),.01)[0],'C')

    def test_bad_profiles_rejected(self):
        for value in ([], [[0]*128], [[float('nan')]*128], [[1]*12]):
            with self.subTest(value=str(value)[:20]),self.assertRaises(ValueError):
                Registry([MarkerProfile('bad',value,.9)])
        with self.assertRaises(ValueError):
            Registry([MarkerProfile('A',[vector(0).tolist()],1.2)])

    def test_duplicate_names_rejected_after_trimming(self):
        with self.assertRaises(ValueError):
            Registry([MarkerProfile('A',[vector(0).tolist()],.9),MarkerProfile(' A ',[vector(1).tolist()],.9)])

    def test_malformed_density_reports_validation_error(self):
        for stats in ({'mean':['bad']*128,'var':[1.]*128,'ll_thr':0.},
                      {'mean':[0.]*128,'var':[1.]*128,'ll_thr':'bad'},
                      {'mean':None,'var':[1.]*128,'ll_thr':0.}):
            with self.subTest(stats=stats), self.assertRaises(ValueError):
                Registry([MarkerProfile('A',[vector(0).tolist()],.9,**stats)])

    def test_atomic_save_backup_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'registry.json'
            registry(1).save(path)
            registry(4).save(path)
            self.assertEqual(len(Registry.load(path).markers),4)
            self.assertEqual(len(Registry.load(str(path)+'.bak').markers),1)
            r=registry()
            r.markers[0].thr=float('nan')
            with self.assertRaises(ValueError):
                r.save(path)
            self.assertEqual(len(Registry.load(path).markers),4)

    def test_model_mismatch_rejected(self):
        r=registry(1)
        r.markers[0].embedder_sha256='first'
        self.assertEqual(Matcher(r,'second').candidate(vector(0),.01)[3],'model_mismatch')

    def test_ambiguous_and_unenrolled_rejected(self):
        r=registry(2)
        r.markers[1].proto=[vector(0).tolist()]
        self.assertEqual(Matcher(r).candidate(vector(0),.01)[3],'ambiguous')
        self.assertEqual(Matcher(registry()).candidate(vector(5),.01)[3],'below_threshold')


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.manager=IdentityManager(registry(),Settings())

    def confirm(self,observations,start=0.):
        for i in range(3):
            self.manager.update(observations,start+i*.1)

    def test_four_concurrent_identities(self):
        self.confirm({i:vector(i) for i in range(4)})
        self.assertEqual([s.name for s in self.manager.states.values()],['A','B','C','D'])

    def test_unknown_expires_instead_of_resetting_streak(self):
        self.confirm({1:vector(0)})
        for now in (.3,.4,.5,.6,.7,.8,.9):
            self.manager.update({1:vector(7)},now)
        self.assertEqual(self.manager.states[1].name,'unknown')

    def test_quality_holds_briefly_then_expires(self):
        self.confirm({1:vector(0)})
        self.manager.update({1:None},.4)
        self.assertEqual(self.manager.states[1].name,'A')
        self.manager.update({1:None},.9)
        self.assertEqual(self.manager.states[1].name,'unknown')

    def test_identity_can_correct_after_perfect_old_score(self):
        self.confirm({1:vector(0)})
        self.confirm({1:vector(1)},.3)
        self.assertEqual(self.manager.states[1].name,'B')

    def test_crossed_track_ids_recover_without_duplicate_names(self):
        self.confirm({1:vector(0),2:vector(1)})
        for now in (.3,.4,.5):
            self.manager.update({1:vector(1),2:vector(0)},now)
            names=[s.name for s in self.manager.states.values() if s.name!='unknown']
            self.assertEqual(len(names),len(set(names)))
        self.assertEqual(self.manager.states[1].name,'B')
        self.assertEqual(self.manager.states[2].name,'A')

    def test_duplicate_evidence_does_not_invent_second_choice(self):
        self.confirm({1:vector(0),2:vector(0)})
        self.assertTrue(all(s.name=='unknown' for s in self.manager.states.values()))

    def test_reentry_new_track_supersedes_absent_owner(self):
        self.confirm({1:vector(0)})
        self.confirm({2:vector(0)},.3)
        self.assertEqual(self.manager.states[2].name,'A')
        self.assertEqual(self.manager.states[1].name,'unknown')

    def test_no_detections_expires_name_and_state(self):
        self.confirm({1:vector(0)})
        self.manager.update({},.9)
        self.assertEqual(self.manager.states[1].name,'unknown')
        self.manager.update({},4.)
        self.assertFalse(self.manager.states)

    def test_confirmation_not_accumulated_over_long_gaps(self):
        for now in (0.,4.,8.):
            self.manager.update({1:vector(0)},now)
        self.assertEqual(self.manager.states[1].name,'unknown')

    def test_slow_but_consistent_frames_can_confirm(self):
        for now in (0.,.8,1.6):
            self.manager.update({1:vector(0)},now)
        self.assertEqual(self.manager.states[1].name,'A')


class EnrollmentTests(unittest.TestCase):
    def test_empty_and_partial_enrollment_do_not_save(self):
        e=Enrollment('A')
        with self.assertRaises(ValueError):
            e.build()
        e.add(vector(0),0)
        with self.assertRaises(ValueError):
            e.advance()

    def test_all_views_required_and_profile_roundtrip(self):
        e=Enrollment('A',6,.1)
        now=0.
        for step in range(len(VIEWS)):
            for _ in range(6):
                now+=.2
                self.assertTrue(e.add(vector(0),now))
            e.advance()
        self.assertTrue(e.complete)
        p=e.build('hash')
        self.assertEqual(p.enroll_used,30)
        self.assertEqual(p.calibration,'within_session_temporal_holdout')
        self.assertEqual(Matcher(Registry([p]),'hash').candidate(vector(0),.01)[0],'A')

    def test_sample_interval_limits_duplicate_frames(self):
        e=Enrollment('A')
        self.assertTrue(e.add(vector(0),1))
        self.assertFalse(e.add(vector(0),1.01))

    def test_quality_gating_reports_reason(self):
        image=np.zeros((100,100,3),dtype=np.uint8)
        self.assertEqual(crop_quality(image,[0,0,10,10],Settings())[1],'too_small')
        self.assertEqual(crop_quality(image,[0,0,90,90],Settings())[1],'blurred')

    def test_configuration_rejects_incompatible_recovery_thresholds(self):
        with self.assertRaises(ValueError):
            replace(Settings(),detector_confidence=.5).validate()


if __name__=='__main__':
    unittest.main()
