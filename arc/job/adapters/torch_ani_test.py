#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.torchani module
"""

import os
import shutil
import unittest

from arc.common import ARC_PATH, almost_equal_lists, read_yaml_file
from arc.job.adapters.torch_ani import TorchANIAdapter
from arc.species import ARCSpecies
from arc.species.vectors import calculate_distance, calculate_angle, calculate_dihedral_angle


class TestTorchANIAdapter(unittest.TestCase):
    """
    Contains unit tests for the TorchANIAdapter class.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.job_1 = TorchANIAdapter(execution_type='incore',
                                    job_type='sp',
                                    project='test_1',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_1'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    )
        cls.job_2 = TorchANIAdapter(execution_type='incore',
                                    job_type='opt',
                                    project='test_2',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_2'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    )
        etoh_xyz = """ C       -0.93790674    0.28066443    0.10572942
                       C        0.35659906   -0.44954997    0.05020174
                       O        0.36626530   -1.59397979   -0.38012632
                       H       -1.68923915   -0.33332195    0.61329151
                       H       -0.85532021    1.23909997    0.62578027
                       H       -1.30704889    0.46001151   -0.90948878
                       H        0.76281007   -0.50036590    1.06483009
                       H        1.04287051    0.12137561   -0.58236096
                       H        1.27820018   -1.93031032   -0.35203473"""
        cls.job_3 = TorchANIAdapter(execution_type='incore',
                                    job_type='opt',
                                    project='test_3',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_3'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    constraints=[([1, 2, 3], 109), ([2, 3], 1.4), [(3, 2, 1, 5), 179.8]],
                                    )
        cls.job_4 = TorchANIAdapter(execution_type='incore',
                                    job_type='opt',
                                    project='test_4',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_4'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    )
        cls.job_5 = TorchANIAdapter(execution_type='incore',
                                    job_type='sp',
                                    project='test_5',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_5'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    )
        cls.job_6 = TorchANIAdapter(execution_type='incore',
                                    job_type='freq',
                                    project='test_6',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_6'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    )
        cls.job_7 = TorchANIAdapter(execution_type='incore',
                                    job_type='optfreq',
                                    project='test_7',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_7'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                    )
        cls.job_8 = TorchANIAdapter(execution_type='incore',
                                    job_type='sp',
                                    project='test_8',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_8'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    )
        cls.job_9 = TorchANIAdapter(execution_type='incore',
                                    job_type='force',
                                    project='test_1',
                                    project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_1'),
                                    species=[ARCSpecies(label='EtOH', smiles='CCO')],
                                    )
        cls.job_10 = TorchANIAdapter(execution_type='incore',
                                     job_type='freq',
                                     project='test_5',
                                     project_directory=os.path.join(ARC_PATH, 'arc', 'testing', 'test_TorchANIAdapter_5'),
                                     species=[ARCSpecies(label='EtOH', smiles='CCO', xyz=etoh_xyz)],
                                     )

    def test_run_sp(self):
        """Test the run_sp() method"""
        self.assertIsNone(self.job_8.sp)
        self.job_8.execute()
        self.assertAlmostEqual(self.job_8.sp, -406925.6663602, places=3)

    def test_run_force(self):
        """Test the run_force() method."""
        self.job_9.execute()
        self.assertTrue(almost_equal_lists(self.job_9.force,
                                           [[0.0016908727120608091, 0.009869818575680256, 0.0010390167590230703],
                                            [0.008561883121728897, 0.013575542718172073, 0.0018425204325467348],
                                            [0.010151226073503494, 0.0019111409783363342, 0.0008542370051145554],
                                            [0.0007229172624647617, -0.0034123272635042667, 0.0017430195584893227],
                                            [0.0009874517563730478, -0.0030386836733669043, -0.002235014922916889],
                                            [-0.0011509160976856947, -0.00131673039868474, -0.00019995146431028843],
                                            [-0.003339727409183979, -0.012097489088773727, 0.0027393437922000885],
                                            [-0.0028020991012454033, -0.011338669806718826, -0.005346616730093956],
                                            [-0.014821597374975681, 0.005847393535077572, -0.0004365567583590746]],
                                           rtol=1e-3, atol=1e-5))

    def test_run_opt(self):
        """Test the run_opt() method."""
        self.assertIsNone(self.job_2.opt_xyz)
        self.job_2.execute()
        self.assertEqual(self.job_2.opt_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'))

        self.assertIsNone(self.job_3.opt_xyz)
        self.job_3.execute()
        self.assertAlmostEqual(calculate_distance(coords=self.job_3.opt_xyz['coords'], atoms=[2, 3], index=1), 1.4, places=2)
        self.assertAlmostEqual(calculate_angle(coords=self.job_3.opt_xyz['coords'], atoms=[1, 2, 3], index=1), 109, places=2)
        self.assertAlmostEqual(calculate_dihedral_angle(coords=self.job_3.opt_xyz['coords'], torsion=[3, 2, 1, 5], index=1),
                               179.8, places=2)

        self.assertIsNone(self.job_4.opt_xyz)
        self.job_4.execute()
        self.assertEqual(self.job_4.opt_xyz['symbols'], ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H', 'H'))

    def test_run_vibrational_analysis(self):
        """Test the run_vibrational_analysis() method."""
        self.job_10.execute()
        results = read_yaml_file(path=self.job_10.local_path_to_output_file)
        self.assertEqual(list(results.keys()), ['force_constants', 'freqs', 'hessian', 'modes', 'reduced_masses'])
        self.assertGreater(len(results['force_constants']), 0)
        self.assertGreater(len(results['freqs']), 0)
        self.assertGreater(len(results['hessian'][0][0]), 0)
        self.assertGreater(len(results['modes'][0][0]), 0)
        self.assertGreater(len(results['reduced_masses']), 0)

    def test_run_freq(self):
        """Test the run_freq() method."""
        self.assertIsNone(self.job_6.freqs)
        self.assertIsNone(self.job_7.freqs)
        self.job_6.execute()
        self.job_7.execute()
        self.assertAlmostEqual(self.job_6.freqs[-1], 3756.57643, places=3)
        self.assertGreater(self.job_7.freqs[-1], 3800)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        for i in range(10):
            path = os.path.join(ARC_PATH, 'arc', 'testing', f'test_TorchANIAdapter_{i}')
            shutil.rmtree(path, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
