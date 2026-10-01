import unittest
from pinch.calibrate import choose_threshold
from pinch.identity import Matcher
from pinch.registry import Registry, MarkerProfile
import numpy as np


class CalibrationTests(unittest.TestCase):
    def test_requires_positive_and_negative_samples(self):
        self.assertIsNone(choose_threshold([.99]*20,[]))
        self.assertIsNone(choose_threshold([.99]*19,[.2]*20))

    def test_selected_threshold_rejects_negative_boundary(self):
        result=choose_threshold([.99]*20,[.8]*20,0.)
        self.assertGreater(result['threshold'],.8)
        self.assertEqual(result['negative_acceptance'],0.)
        self.assertEqual(result['positive_acceptance'],1.)
        p=[1.]+[0.]*127
        matcher=Matcher(Registry([MarkerProfile('A',[p],result['threshold'])]))
        # Test the same numeric scores as calibration, including its boundary.
        self.assertEqual(matcher.candidate(p,.01,np.array([.8]))[3],'below_threshold')

    def test_inseparable_scores_report_no_safe_threshold(self):
        self.assertIsNone(choose_threshold([1.]*20,[1.]*20,0.))
