#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests for the arc.checks.ts module
"""

import unittest
import os
import shutil

import numpy as np

import arc.checks.ts as ts
import arc.rmgdb as rmgdb
from arc.common import ARC_PATH
from arc.job.factory import job_factory
from arc.level import Level
from arc.parser import parse_normal_mode_displacement
from arc.reaction import ARCReaction
from arc.species.species import ARCSpecies, TSGuess


class TestChecks(unittest.TestCase):
    """
    Contains unit tests for the check module.
    """
    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = rmgdb.make_rmg_database_object()
        rmgdb.load_families_only(cls.rmgdb)
        cls.rms_list_1 = [0.01414213562373095, 0.05, 0.04, 0.5632938842203065, 0.7993122043357026, 0.08944271909999159,
                          0.10677078252031312, 0.09000000000000001, 0.05, 0.09433981132056604]
        path_1 = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'freq', 'C3H7_intra_h_TS.out')
        cls.freqs_1, cls.normal_modes_disp_1 = parse_normal_mode_displacement(path_1)
        cls.ts_1 = ARCSpecies(label='TS', is_ts=True)
        cls.ts_1.ts_guesses = [TSGuess(family='intra_H_migration', xyz='C 0 0 0'),
                               TSGuess(family='intra_H_migration', xyz='C 0 0 0'),
                               ]
        cls.ts_xyz_1 = """O      -0.63023600    0.92494700    0.43958200
C       0.14513500   -0.07880000   -0.04196400
C      -0.97050300   -1.02992900   -1.65916600
N      -0.75664700   -2.16458700   -1.81286400
H      -1.25079800    0.57954500    1.08412300
H       0.98208300    0.28882200   -0.62114100
H       0.30969500   -0.94370100    0.59100600
H      -1.47626400   -0.10694600   -1.88883800"""  # 'N#[CH].[CH2][OH]'

        cls.ts_xyz_2 = """C       0.52123900   -0.93806900   -0.55301700
C       0.15387500    0.18173100    0.37122900
C      -0.89554000    1.16840700   -0.01362800
H       0.33997700    0.06424800    1.44287100
H       1.49602200   -1.37860200   -0.29763200
H       0.57221700   -0.59290500   -1.59850500
H       0.39006800    1.39857900   -0.01389600
H      -0.23302200   -1.74751100   -0.52205400
H      -1.43670700    1.71248300    0.76258900
H      -1.32791000    1.11410600   -1.01554900"""  # C[CH]C <=> [CH2]CC
        cls.r_xyz_2a = """C                  0.50180491   -0.93942231   -0.57086745
C                  0.01278145    0.13148427    0.42191407
C                 -0.86874485    1.29377369   -0.07163907
H                  0.28549447    0.06799101    1.45462711
H                  1.44553946   -1.32386345   -0.24456986
H                  0.61096295   -0.50262210   -1.54153222
H                 -0.24653265    2.11136864   -0.37045418
H                 -0.21131163   -1.73585284   -0.61629002
H                 -1.51770930    1.60958621    0.71830245
H                 -1.45448167    0.96793094   -0.90568876"""
        cls.r_xyz_2b = """C                  0.50180491   -0.93942231   -0.57086745
C                  0.01278145    0.13148427    0.42191407
H                  0.28549447    0.06799101    1.45462711
H                  1.44553946   -1.32386345   -0.24456986
H                  0.61096295   -0.50262210   -1.54153222
H                 -0.24653265    2.11136864   -0.37045418
C                 -0.86874485    1.29377369   -0.07163907
H                 -0.21131163   -1.73585284   -0.61629002
H                 -1.51770930    1.60958621    0.71830245
H                 -1.45448167    0.96793094   -0.90568876"""
        cls.p_xyz_2 = """C                  0.48818717   -0.94549701   -0.55196729
C                  0.35993708    0.29146456    0.35637075
C                 -0.91834764    1.06777042   -0.01096751
H                  0.30640232   -0.02058840    1.37845537
H                  1.37634603   -1.48487836   -0.29673876
H                  0.54172192   -0.63344406   -1.57405191
H                  1.21252186    0.92358349    0.22063264
H                 -0.36439762   -1.57761595   -0.41622918
H                 -1.43807526    1.62776079    0.73816131
H                 -1.28677889    1.04716138   -1.01532486"""
        cls.ts_spc_2 = ARCSpecies(label='TS', is_ts=True, xyz=cls.ts_xyz_2)
        cls.ts_spc_2.mol_from_xyz()
        cls.reactant_2a = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=cls.r_xyz_2a)
        cls.reactant_2b = ARCSpecies(label='C[CH]C', smiles='C[CH]C', xyz=cls.r_xyz_2b)  # same as a, only once C atom shifted place in the reactant xyz
        cls.product_2 = ARCSpecies(label='[CH2]CC', smiles='[CH2]CC', xyz=cls.p_xyz_2)
        cls.rxn_2a = ARCReaction(r_species=[cls.reactant_2a], p_species=[cls.product_2])
        cls.rxn_2a.ts_species = cls.ts_spc_2
        cls.rxn_2b = ARCReaction(r_species=[cls.reactant_2b], p_species=[cls.product_2])
        cls.rxn_2b.ts_species = cls.ts_spc_2

    def test_check_ts_energy(self):
        """Test the check_ts_energy() method"""
        def populate_ts_checks_and_check_ts_energy(reaction: ARCReaction, parameter='E0'):
            """A helper function for running populate_ts_checks() and check_ts_energy()"""
            reaction.ts_species.populate_ts_checks()
            ts.check_ts_energy(reaction=reaction, parameter=parameter)

        rxn1 = ARCReaction(r_species=[ARCSpecies(label='s1', smiles='C')], p_species=[ARCSpecies(label='s2', smiles='C')])
        rxn1.ts_species = ARCSpecies(label='TS', is_ts=True)
        # no data
        populate_ts_checks_and_check_ts_energy(rxn1)
        self.assertTrue(rxn1.ts_species.ts_checks['e_elect'])
        # only E0 (correct)
        rxn1.r_species[0].e0 = 2
        rxn1.p_species[0].e0 = 50
        rxn1.ts_species.e0 = 100
        populate_ts_checks_and_check_ts_energy(rxn1)
        self.assertTrue(rxn1.ts_species.ts_checks['E0'])
        # only E0 (incorrect)
        rxn1.r_species[0].e0 = 2
        rxn1.p_species[0].e0 = 50
        rxn1.ts_species.e0 = -100
        populate_ts_checks_and_check_ts_energy(rxn1)
        self.assertFalse(rxn1.ts_species.ts_checks['E0'])
        # only E0 (partial data)
        rxn1.r_species[0].e0 = 2
        rxn1.p_species[0].e0 = None
        rxn1.ts_species.e0 = -100
        populate_ts_checks_and_check_ts_energy(rxn1)
        self.assertTrue(rxn1.ts_species.ts_checks['e_elect'])
        # also e_elect (correct)
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = 50
        rxn1.ts_species.e_elect = 100
        populate_ts_checks_and_check_ts_energy(rxn1)
        self.assertTrue(rxn1.ts_species.ts_checks['e_elect'])
        # also e_elect (incorrect)
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = 50
        rxn1.ts_species.e_elect = -100
        populate_ts_checks_and_check_ts_energy(rxn1)
        self.assertFalse(rxn1.ts_species.ts_checks['e_elect'])
        # also e_elect (partial data)
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = None
        rxn1.ts_species.e_elect = -100
        populate_ts_checks_and_check_ts_energy(rxn1)
        self.assertTrue(rxn1.ts_species.ts_checks['e_elect'])
        # check e_elect directly (correct)
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = 50
        rxn1.ts_species.e_elect = 100
        populate_ts_checks_and_check_ts_energy(rxn1, parameter='e_elect')
        self.assertTrue(rxn1.ts_species.ts_checks['e_elect'])
        # check e_elect directly (incorrect)
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = 50
        rxn1.ts_species.e_elect = -100
        populate_ts_checks_and_check_ts_energy(rxn1, parameter='e_elect')
        self.assertFalse(rxn1.ts_species.ts_checks['e_elect'])
        # check e_elect directly (partial data)
        rxn1.r_species[0].e_elect = 2
        rxn1.p_species[0].e_elect = 50
        rxn1.ts_species.e_elect = None
        populate_ts_checks_and_check_ts_energy(rxn1, parameter='e_elect')
        self.assertTrue(rxn1.ts_species.ts_checks['e_elect'])

    # def test_check_normal_mode_displacement(self):
    #     rxn1 = ARCReaction(r_species=[ARCSpecies(label='[CH2]CC', smiles='[CH2]CC')],
    #                        p_species=[ARCSpecies(label='C[CH]C', smiles='C[CH]C')])
    #     rxn1.ts_species = ARCSpecies(label='TS1', is_ts=True)
    #     rxn1.ts_species.populate_ts_checks()
    #     self.assertFalse(rxn1.ts_species.ts_checks['normal_mode_displacement'])
    #
    #     rxn1.determine_family(rmg_database=self.rmgdb)
    #     ts.check_normal_mode_displacement(reaction=rxn1, rxn_zone_atom_indices=[15, 25])
    #     self.assertFalse(rxn1.ts_species.ts_checks['normal_mode_displacement'])
    #
    #     rxn1.ts_species.populate_ts_checks()
    #     ts.check_normal_mode_displacement(reaction=rxn1, rxn_zone_atom_indices=[1, 2, 3])
    #     self.assertTrue(rxn1.ts_species.ts_checks['normal_mode_displacement'])

    def test_check_ts_freq_job(self):
        """Test the check_ts_freq_job() function"""
        self.reactant_2a.e0, self.product_2.e0, self.ts_spc_2.e0 = 0, 10, 100
        rxn = ARCReaction(r_species=[self.reactant_2a], p_species=[self.product_2])
        rxn.ts_species = self.ts_spc_2
        rxn.determine_family(self.rmgdb)
        job = job_factory(job_adapter='gaussian',
                          species=[self.ts_spc_2],
                          job_type='composite',
                          level=Level(method='CBS-QB3'),
                          project='test_project',
                          project_directory=os.path.join(ARC_PATH,
                                                         'Projects',
                                                         'arc_project_for_testing_delete_after_usage4'),
                          )
        job.local_path_to_output_file = os.path.join(ts.ARC_PATH, 'arc', 'testing', 'composite',
                                                     'TS_intra_H_migration_CBS-QB3.out')
        # print(ts_spc.ts_checks)
        # switch_ts = ts.check_ts_freq_job(species=ts_spc, reaction=rxn, job=job)
        # print(switch_ts)
        # print(ts_spc.ts_checks)
        # raise

    def test_invalidate_rotors_with_both_pivots_in_a_reactive_zone(self):
        """Test the invalidate_rotors_with_both_pivots_in_a_reactive_zone() function"""
        ts_spc_1 = ARCSpecies(label='TS', is_ts=True, xyz=self.ts_xyz_1)
        ts_spc_1.mol_from_xyz()
        ts_spc_1.determine_rotors()
        # Manually add the rotor that breaks the TS, it is not identified automatically.
        ts_spc_1.rotors_dict[1] = {'pivots': [2, 3],
                                   'top': [4, 8],
                                   'scan': [1, 2, 3, 4],
                                   'torsion': [0, 1, 2, 3],
                                   'success': None,
                                   'invalidation_reason': '',
                                   'dimensions': 1}
        rxn_zone_atom_indices = [1, 2]
        ts.invalidate_rotors_with_both_pivots_in_a_reactive_zone(species=ts_spc_1,
                                                                 rxn_zone_atom_indices=rxn_zone_atom_indices)
        self.assertEqual(ts_spc_1.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(ts_spc_1.rotors_dict[0]['invalidation_reason'], '')
        self.assertIsNone(ts_spc_1.rotors_dict[0]['success'])
        self.assertEqual(ts_spc_1.rotors_dict[1]['pivots'], [2, 3])
        self.assertEqual(ts_spc_1.rotors_dict[1]['scan'], [1, 2, 3, 4])
        self.assertEqual(ts_spc_1.rotors_dict[1]['invalidation_reason'],
                         'Pivots participate in the TS reaction zone (code: pivTS). ')
        self.assertEqual(ts_spc_1.rotors_dict[1]['success'], False)

        ts_spc_2 = ARCSpecies(label='TS', is_ts=True, xyz=self.ts_xyz_2)
        ts_spc_2.mol_from_xyz()
        rxn_zone_atom_indices = [1, 2, 6]
        ts.invalidate_rotors_with_both_pivots_in_a_reactive_zone(species=ts_spc_2,
                                                                 rxn_zone_atom_indices=rxn_zone_atom_indices)
        self.assertEqual(ts_spc_2.rotors_dict[0]['pivots'], [1, 2])
        self.assertEqual(ts_spc_2.rotors_dict[0]['scan'], [5, 1, 2, 3])
        self.assertEqual(ts_spc_2.rotors_dict[0]['invalidation_reason'], '')
        self.assertIsNone(ts_spc_2.rotors_dict[0]['success'])
        self.assertEqual(ts_spc_2.rotors_dict[1]['pivots'], [2, 3])
        self.assertEqual(ts_spc_2.rotors_dict[1]['scan'], [1, 2, 3, 9])
        self.assertEqual(ts_spc_2.rotors_dict[1]['invalidation_reason'],
                         'Pivots participate in the TS reaction zone (code: pivTS). ')
        self.assertEqual(ts_spc_2.rotors_dict[1]['success'], False)

    def test_get_indices_of_atoms_participating_in_reaction(self):
        """Test the get_indices_of_atoms_participating_in_reaction() function"""
        # Todo - check again
        self.assertEqual(ts.get_indices_of_atoms_participating_in_reaction(normal_mode_disp=self.normal_modes_disp_1,
                                                                           freqs=self.freqs_1,
                                                                           ts_guesses=self.ts_1.ts_guesses,
                                                                           ), [3, 0, 1])

    def test_get_rms_from_normal_modes_disp(self):
        """Test the get_rms_from_normal_modes_disp() function"""
        rms = ts.get_rms_from_normal_mode_disp(self.normal_modes_disp_1, np.array([-1000.3, 320.5], np.float64))
        self.assertEqual(rms, [0.07874007874011811,
                               0.07280109889280519,
                               0.0,
                               0.9914635646356349,
                               0.03605551275463989,
                               0.034641016151377546,
                               0.0,
                               0.033166247903554,
                               0.01414213562373095,
                               0.0],
                         )

    def test_get_index_of_abs_largest_neg_freq(self):
        """Test the get_index_of_abs_largest_neg_freq() function"""
        self.assertIsNone(ts.get_index_of_abs_largest_neg_freq(np.array([], np.float64)))
        self.assertIsNone(ts.get_index_of_abs_largest_neg_freq(np.array([1, 320.5], np.float64)))
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([-1], np.float64)), 0)
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([-1, 320.5], np.float64)), 0)
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([320.5, -1], np.float64)), 1)
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([320.5, -1, -80, -90, 5000], np.float64)), 3)
        self.assertEqual(ts.get_index_of_abs_largest_neg_freq(np.array([-320.5, -1, -80, -90, 5000], np.float64)), 0)

    def test_get_expected_num_atoms_with_largest_normal_mode_disp(self):
        """Test the get_expected_num_atoms_with_largest_normal_mode_disp() function"""
        normal_disp_mode_rms = [0.01414213562373095, 0.05, 0.04, 0.5632938842203065, 0.7993122043357026,
                                0.08944271909999159, 0.10677078252031312, 0.09000000000000001, 0.05, 0.09433981132056604]
        num_of_atoms = ts.get_expected_num_atoms_with_largest_normal_mode_disp(
            normal_disp_mode_rms=normal_disp_mode_rms,
            ts_guesses=self.ts_1.ts_guesses)
        self.assertEqual(num_of_atoms, 4)

    def test_get_rxn_normal_mode_disp_atom_number(self):
        """Test the get_rxn_normal_mode_disp_atom_number function"""
        with self.assertRaises(TypeError):
            ts.get_rxn_normal_mode_disp_atom_number(15)
        with self.assertRaises(TypeError):
            ts.get_rxn_normal_mode_disp_atom_number('family', 'family')
        with self.assertRaises(TypeError):
            ts.get_rxn_normal_mode_disp_atom_number('family', ['family'])
        with self.assertRaises(TypeError):
            ts.get_rxn_normal_mode_disp_atom_number('family', 15.215)
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number('default'), 3)
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number('intra_H_migration'), 3)
        self.assertEqual(ts.get_rxn_normal_mode_disp_atom_number('intra_H_migration', rms_list=self.rms_list_1), 4)

    def test_find_equivalent_atoms_in_reactants_and_products(self):
        """Test the find_equivalent_atoms_in_reactants_and_products() function"""
        # Calling find_equivalent_atoms_in_reactants() also determined the family, important for additional unit tests.
        equivalence_map_1 = ts.find_equivalent_atoms_in_reactants(self.rxn_2a)
        # Both C 0 and C 2 are equivalent, C 1 is unique, and H 4-9 are equivalent as well.
        self.assertEqual(equivalence_map_1, [[0, 2], [1], [4, 5, 6, 7, 8, 9]])
        equivalence_map_2 = ts.find_equivalent_atoms_in_reactants(self.rxn_2b)
        self.assertEqual(equivalence_map_2, [[0, 6], [1], [3, 4, 5, 7, 8, 9]])

    def test_get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(self):
        """Test the get_atom_indices_of_labeled_atoms_in_an_rmg_reaction() function"""
        for atom, symbol in zip(self.rxn_2a.r_species[0].mol.atoms, ['C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H', 'H']):
            self.assertEqual(atom.symbol, symbol)
        rmg_reactions = self.rxn_2a.family.generate_reactions(reactants=[spc.mol for spc in self.rxn_2a.r_species],
                                                              products=[spc.mol for spc in self.rxn_2a.p_species],
                                                              prod_resonance=True,
                                                              delete_labels=False,
                                                              )
        map = ts.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(rmg_reactions[0])
        self.assertEqual(map, {'*2': 0, '*1': 1, '*3': 4})

        for atom, symbol in zip(self.rxn_2b.r_species[0].mol.atoms, ['C', 'C', 'H', 'H', 'H', 'H', 'C', 'H', 'H', 'H']):
            self.assertEqual(atom.symbol, symbol)
        rmg_reactions = self.rxn_2b.family.generate_reactions(reactants=[spc.mol for spc in self.rxn_2b.r_species],
                                                              products=[spc.mol for spc in self.rxn_2b.p_species],
                                                              prod_resonance=True,
                                                              delete_labels=False,
                                                              )
        map = ts.get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(rmg_reactions[0])
        self.assertEqual(map, {'*2': 0, '*1': 1, '*3': 3})

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests
        """
        projects = ['arc_project_for_testing_delete_after_usage4']
        for project in projects:
            project_directory = os.path.join(ARC_PATH, 'Projects', project)
            shutil.rmtree(project_directory, ignore_errors=True)


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
