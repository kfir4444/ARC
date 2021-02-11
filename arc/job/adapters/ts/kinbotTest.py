#!/usr/bin/env python3
# encoding: utf-8

"""
This module contains unit tests of the arc.job.adapters.ts.autotst module
"""

import os
import shutil
import unittest

from rmgpy.reaction import Reaction
from rmgpy.species import Species

from arc.common import arc_path
from arc.job.adapters.ts.kinbot_ts import KinBotAdapter
from arc.reaction import ARCReaction
from arc.rmgdb import make_rmg_database_object, load_families_only


class TestKinBotAdapter(unittest.TestCase):
    """
    Contains unit tests for the AutoTSTAdapter class.
    """

    @classmethod
    def setUpClass(cls):
        """
        A method that is run before all unit tests in this class.
        """
        cls.maxDiff = None
        cls.rmgdb = make_rmg_database_object()
        load_families_only(cls.rmgdb)

    def test_intra_h_migration(self):
        """Test KinBot for intra H migration reactions"""
        rxn1 = ARCReaction(reactants=['CC[O]'], products=['[CH2]CO'])
        rxn1.rmg_reaction = Reaction(reactants=[Species().from_smiles('CC[O]')],
                                     products=[Species().from_smiles('[CH2]CO')])
        rxn1.determine_family(rmg_database=self.rmgdb)
        rxn1.determine_rxn_charge()
        rxn1.determine_rxn_multiplicity()
        rxn1.arc_species_from_rmg_reaction()
        self.assertEqual(rxn1.family.label, 'intra_H_migration')
        kinbot1 = KinBotAdapter(reactions=[rxn1],
                                testing=True,
                                project='test',
                                project_directory=os.path.join(arc_path, 'arc', 'testing', 'test_KinBot', 'tst1'),
                                )
        kinbot1.execute_incore()
        self.assertTrue(rxn1.ts_species.is_ts)
        self.assertEqual(rxn1.ts_species.charge, 0)
        self.assertEqual(rxn1.ts_species.multiplicity, 2)
        self.assertEqual(len(rxn1.ts_species.ts_guesses), 4)
        self.assertEqual(rxn1.ts_species.ts_guesses[0].initial_xyz['symbols'],
                         ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(rxn1.ts_species.ts_guesses[1].initial_xyz['symbols'],
                         ('C', 'C', 'O', 'H', 'H', 'H', 'H', 'H'))
        self.assertEqual(len(rxn1.ts_species.ts_guesses[1].initial_xyz['coords']), 8)
        self.assertEqual(len(rxn1.ts_species.ts_guesses[2].initial_xyz['coords']), 8)
        self.assertEqual(len(rxn1.ts_species.ts_guesses[3].initial_xyz['coords']), 8)

    @classmethod
    def tearDownClass(cls):
        """
        A function that is run ONCE after all unit tests in this class.
        Delete all project directories created during these unit tests.
        """
        shutil.rmtree(os.path.join(arc_path, 'arc', 'testing', 'test_KinBot'))


if __name__ == '__main__':
    unittest.main(testRunner=unittest.TextTestRunner(verbosity=2))
