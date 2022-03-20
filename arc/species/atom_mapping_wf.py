"""
A module for atom-mapping a species or a set of species.

Species atom-map logic:
1. Fingerprint the species
2. Identify and loop superimposable candidate backbone mapping
3. Recursively modify dihedrals until the structures overlap to some tolerance
4. Determine RMSD to backbone, if good then determine RMSD to H's
5. When mapping H's on terminal heavy atoms, check whether rotating this rotor will reduce the overall RMSD
   if there's more than one H on that terminal atom
"""

from collections import deque
from itertools import product
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from qcelemental.exceptions import ValidationError
from qcelemental.models.molecule import Molecule as QCMolecule

from rmgpy.molecule import Molecule
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import convert_list_index_0_to_1, extremum_list, logger
from arc.exceptions import SpeciesError
from arc.species.mapping import get_rmg_reactions_from_arc_reaction, \
    get_atom_indices_of_labeled_atoms_in_an_rmg_reaction
from arc.species import ARCSpecies
from arc.species.conformers import determine_chirality
from arc.species.converter import compare_confs, sort_xyz_using_indices, translate_xyz, xyz_from_data, xyz_to_str
from arc.species.vectors import calculate_angle, calculate_dihedral_angle, calculate_distance, get_delta_angle

from numpy import unique

if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import TemplateReaction
    from rmgpy.data.rmg import RMGDatabase
    from rmgpy.molecule.molecule import Atom
    from rmgpy.reaction import Reaction
    from arc.reaction import ARCReaction

RESERVED_FINGERPRINT_KEYS = ['self', 'chirality', 'label']


def set_up_for_map_reaction(rxn: 'ARCReaction',
                            backend: str = 'ARC',
                            db: Optional['RMGDatabase'] = None,
                            ) -> Optional[List[int]]:
    """
    Classify reaction using reaction family rules, sends them to correct mapping functions.

    Args:
        rxn (ARCReaction): An ARCReaction object instance.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """

    pass


def map_rxn(rxn: 'ARCReaction',
            backend: str = 'ARC',
            db: Optional['RMGDatabase'] = None,
            ) -> Optional[List[int]]:
    """
    A wrapper function for mapping reaction, uses databases for mapping with the correct reaction family parameters.

    Strategy:
        0) Find out the reaction family class that the reaction is associated with, done before(?).
        1) Mark atom labels.
        2) get_rmg_reactions_from_arc_reaction, get_atom_indices_of_labeled_atoms_in_an_rmg_reaction.
        3) (For bimolecular reactions) Find the species in which the bond is broken.
        4) Scissor the reactant(s) and product(s).
        4.5) Match pair species.
        5) Map_two_species.
        6) Join maps together.


    Args:
        rxn (ARCReaction): An ARCReaction object instance that belongs to the RMG H_Abstraction reaction family.
        backend (str, optional): Whether to use ``'QCElemental'`` or ``ARC``'s method as the backend.
        db (RMGDatabase, optional): The RMG database instance.

    Returns:
        Optional[List[int]]:
            Entry indices are running atom indices of the reactants,
            corresponding entry values are running atom indices of the products.
    """
    # step 1:
    #    atom_labels = find_atom_labels(rxn,db) #Don't sure if we need that yet, probably not

    # step 2:
    rmg_reactions = get_rmg_reactions_from_arc_reaction(arc_reaction=rxn, backend=backend)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])

    # step 3:
    reactants, products,loc_r,loc_p = prepare_reactants_and_products_for_scissors(rxn, r_label_dict, p_label_dict)

    r_cuts, p_cuts = cut_species_for_mapping(rxn,reactants, products,loc_r,loc_p)

    pairs_of_reactant_and_products = pairing_reactants_and_products_for_mapping(r_cuts, p_cuts)


    # step 5:

    pass


def prepare_reactants_and_products_for_scissors(rxn: 'ARCReaction',
                                                r_label_dict: dict,
                                                p_label_dict: dict,
                                                ) -> Tuple[List[ARCSpecies], List[ARCSpecies], List[int], List[int]]:
    """
    Prepares the species to be scissored.
    Args:
        rxn: ARC reaction object to be mapped
        r_label_dict: the labels of the reactants
        p_label_dict: the labels of the products

    Returns:
            The species in the reactants and products where a bond was broken or formed and the location and number of cutss.
    """

    breaks = []
    forms = []
    actions = rxn.family.forward_recipe.actions
    for action in actions:
        if action[0].lower() == "BREAK_BOND".lower():
            breaks.append(action)
        elif action[0].lower() == "FORM_BOND".lower():
            forms.append(action)

    reactants, products,loc_r,loc_p = list(), list(),[0]*len(rxn.r_species), [0]*len(rxn.p_species)

    for broken_bond in breaks:
        location = 0
        index = 0
        for reactant in rxn.r_species:
            if not r_label_dict[broken_bond[1]] < reactant.number_of_atoms + index:
                location += 1
                index += reactant.number_of_atoms
            else:
                loc_r[location]+=1
                reactants.append(ARCSpecies(label=str(index) + "_reactants",
                                            mol=reactant.mol.copy(deep=True),
                                            xyz=reactant.get_xyz(),
                                            bdes=[(r_label_dict[broken_bond[1]],
                                                   r_label_dict[broken_bond[3]])]))
                break

    for formed_bond in forms:
        location = 0
        index = 0
        for product in rxn.p_species:
            if not p_label_dict[formed_bond[1]] < product.number_of_atoms + index:
                location += 1
                index += product.number_of_atoms
            else:
                loc_p[location] += 1
                products.append(ARCSpecies(label=str(index) + "_products",
                                           mol=product.mol.copy(deep=True),
                                           xyz=product.get_xyz(),
                                           bdes=[(p_label_dict[formed_bond[1]],
                                                  p_label_dict[formed_bond[3]])]))
                break

    return reactants, products,loc_r,loc_p


def cut_species_for_mapping(reactants: List[ARCSpecies],
                            products: List[ARCSpecies],
                            loc_r: List[int],
                            loc_p: List[int],
                            ) -> Optional[Tuple[List[ARCSpecies], List[ARCSpecies]]]
    """
    A function for scissoring the reactants and products, as a preparation for atom mapping.
    Args:
        reactants: A list of the ARCSpecies for scission
        products: A list of the ARCSpecies for scission
        loc_r: A list of the location and number of cuts that is required.
        loc_p: A list of the location and number of cuts that is required.

    Returns:
        A list of scissored reactants and products.
    """
    r_cuts,p_cuts=list(),list()
    for index,reactant in zip(loc_r,reactants):
        if index>0:
            try:
                cuts=reactant.scissors()
                r_cuts+=cuts
            except SpeciesError:
                return None
        else:
            r_cuts+=reactant

    for index,product in zip(loc_p,products):
        if index>0:
            try:
                cuts=product.scissors()
                r_cuts+=cuts
            except SpeciesError:
                return None
        else:
            p_cuts+=product

    return r_cuts,p_cuts


def pairing_reactants_and_products_for_mapping(r_cuts: List[ARCSpecies],
                                               p_cuts: List[ARCSpecies]
                                               )-> List[Tuple[ARCSpecies,ARCSpecies]]:
    """
    A function for matching reactants and products in scissored products.
    Args:
        r_cuts: A list of the scissored species in the reactants
        p_cuts: A list of the scissored species in the reactants

    Returns:
        a list of paired reactant and products, to be sent to map_two_species.

    """
    pass


# ROOH + ROOH <=> RO+ HOH +ROO
# RO OH ROO H <=> RO OH ROO H
# RO RO, OH OH, ROO ROO, H H
