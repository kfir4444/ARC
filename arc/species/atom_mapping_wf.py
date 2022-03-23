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

from rmgpy.molecule import Molecule
from rmgpy.species import Species

import arc.rmgdb as rmgdb
from arc.common import convert_list_index_0_to_1, extremum_list, logger,key_by_val
from arc.exceptions import SpeciesError
from arc.species.mapping import get_rmg_reactions_from_arc_reaction, \
    get_atom_indices_of_labeled_atoms_in_an_rmg_reaction, map_two_species
from arc.species import ARCSpecies
from arc.species.conformers import determine_chirality
from arc.species.converter import compare_confs, sort_xyz_using_indices, translate_xyz, xyz_from_data, xyz_to_str
from arc.species.vectors import calculate_angle, calculate_dihedral_angle, calculate_distance, get_delta_angle


if TYPE_CHECKING:
    from rmgpy.data.kinetics.family import TemplateReaction
    from rmgpy.data.rmg import RMGDatabase
    from rmgpy.molecule.molecule import Atom
    from rmgpy.reaction import Reaction
    from arc.reaction import ARCReaction

RESERVED_FINGERPRINT_KEYS = ['self', 'chirality', 'label']



def map_rxn(rxn: 'ARCReaction',
            backend: str = 'ARC',
            db: Optional['RMGDatabase'] = None,
            ) -> Optional[List[int]]:
    """
    A wrapper function for mapping reaction, uses databases for mapping with the correct reaction family parameters.

    Strategy:
        1) get_rmg_reactions_from_arc_reaction, get_atom_indices_of_labeled_atoms_in_an_rmg_reaction.
        2) (For bimolecular reactions) Find the species in which the bond is broken.
        3) Scissor the reactant(s) and product(s).
        4) Match pair species.
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
    rmg_reactions = get_rmg_reactions_from_arc_reaction(arc_reaction=rxn, backend=backend)
    r_label_dict, p_label_dict = get_atom_indices_of_labeled_atoms_in_an_rmg_reaction(arc_reaction=rxn,
                                                                                      rmg_reaction=rmg_reactions[0])

    # step 2:
    assign_labels_to_products(rxn,p_label_dict)
    reactants, products,loc_r,loc_p = prepare_reactants_and_products_for_scissors(rxn, r_label_dict, p_label_dict)

    #step 3:
    r_cuts, p_cuts = cut_species_for_mapping(rxn,reactants, products,loc_r,loc_p)

    #step 4:
    pairs_of_reactant_and_products = pairing_reactants_and_products_for_mapping(r_cuts, p_cuts)

    # step 5:
    maps = []
    for pair in pairs_of_reactant_and_products:
        maps.append(map_two_species(pair[0],pair[1]))
    pass

    #step 6:
    atom_map = join_map(rxn,maps)

    # Tada! U+1F44F

    return atom_map

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
            The species in the reactants and products where a bond was broken or formed.
    """

    breaks = []
    forms = []
    actions = rxn.family.forward_recipe.actions
    for action in actions:
        if action[0].lower() == "BREAK_BOND".lower():
            breaks.append(action)
        elif action[0].lower() == "FORM_BOND".lower():
            forms.append(action)
    reactants, products, loc_r, loc_p = [0] * len(rxn.r_species), [0] * len(rxn.p_species), [0] * len(rxn.r_species), [
        0] * len(rxn.p_species)
    for broken_bond in breaks:
        location = 0
        index = 0
        for reactant in rxn.r_species:
            if not r_label_dict[broken_bond[1]] < reactant.number_of_atoms + index:
                location += 1
                index += reactant.number_of_atoms
            else:
                if loc_r[location] > 0:
                    loc_r[location] += 1
                    reactants[location].bdes += [(r_label_dict[broken_bond[1]] + 1 - index, r_label_dict[broken_bond[3]] + 1 - index)]
                else:
                    loc_r[location] += 1
                    reactants[location] = ARCSpecies(label="".join(sorted(
                        [key_by_val(r_label_dict, r_label_dict[broken_bond[1]]),
                        key_by_val(p_label_dict, p_label_dict[broken_bond[3]])])),
                    mol = reactant.mol.copy(deep=True),
                    xyz = reactant.get_xyz(),
                    bdes = [(r_label_dict[broken_bond[1]] + 1 - index,
                            r_label_dict[broken_bond[3]] + 1 - index)])

                    for mol1, mol2 in zip(reactants[location].mol.atoms, reactant.mol.atoms):
                        mol1.label = mol2.label
                    break

    for formed_bond in forms:
        location = 0
        index = 0
        for product in rxn.p_species:
            if not p_label_dict[formed_bond[1]] < product.number_of_atoms + index:
                location += 1
                index += product.number_of_atoms
            else:
                if loc_p[location] > 0:
                    loc_p[location]+=1
                    products[location].bdes += [(p_label_dict[formed_bond[1]] + 1 - index, p_label_dict[formed_bond[3]] + 1 - index)]
                else:
                    loc_p[location] += 1
                    products[location] = ARCSpecies(label="".join(sorted(
                        [key_by_val(p_label_dict, p_label_dict[formed_bond[1]]),
                        key_by_val(p_label_dict, p_label_dict[formed_bond[3]])])),
                    mol = product.mol.copy(deep=True),
                    xyz = product.get_xyz(),
                    bdes = [(p_label_dict[formed_bond[1]] + 1 - index,
                            p_label_dict[formed_bond[3]] + 1 - index)])
                    for mol1, mol2 in zip(products[location].mol.atoms, product.mol.atoms):
                        mol1.label = mol2.label
                    break
    for index, value in enumerate(loc_r):
        if value == 0:
            reactants[index] = rxn.r_species[index]

    for index, value in enumerate(loc_p):
        if value == 0:
            products[index] = rxn.p_species[index]

    return reactants, products, loc_r, loc_p
# Add BDE instead of adding species.


def assign_labels_to_products(rxn: 'ARCReaction',
                              p_label_dict: dict):
    """
    Add the indices to the reactants and products.
    Args:
        rxn: ARCReaction object to be mapped
        p_label_dict: the labels of the products

    Returns:
        Adding labels to the atoms of the reactants and products, to be identified later.
    """

    atom_index = 0
    for product in rxn.p_species:
        for atom in product.mol.atoms:
            if atom_index in p_label_dict.values() and (atom.label==str() or atom.label==None):
                atom.label = key_by_val(p_label_dict,atom_index)
            atom_index+=1


def cut_species_for_mapping(reactants: List[ARCSpecies],
                            products: List[ARCSpecies],
                            loc_r: List[int],
                            loc_p: List[int],
                            ) -> Optional[Tuple[List[ARCSpecies], List[ARCSpecies]]]:
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
                reactant.final_xyz=reactant.get_xyz()
                cuts=reactant.scissors()
                r_cuts+=cuts
            except SpeciesError:
                return None
        else:
            r_cuts.append(reactant)

    for index,product in zip(loc_p,products):
        if index>0:
            try:
                product.final_xyz=product.get_xyz()
                cuts=product.scissors()
                p_cuts+=cuts
            except SpeciesError:
                return None
        else:
            p_cuts.append(product)

    return r_cuts,p_cuts


def r_cut_p_cuts_share_adjlist(reactant, product):
    """
    A function for checking if the reactant and product has the same adj list, requires testing for additional varification if it's the required method.
    Args:
        reactant: an ARCSpecies. might be as a result of scissors() 
        product: A list of the scissored species in the reactants

    Returns:
        a list of paired reactant and products, to be sent to map_two_species.

    """
    rmg_1,rmg_2 = Species(),Species()
    rmg_1 = Species(label=reactant.label, molecule=[reactant.mol])
    rmg_2 = Species(label=product.label, molecule=[product.mol])
    return rmg_1.isisomorphic(rmg_2)


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
    pairs = []
    for reactant_cut in r_cuts:
        for product_cut in p_cuts:
            if r_cut_p_cuts_share_adjlist(reactant_cut,product_cut):
                pairs.append((r_cuts,product_cut))
                p_cuts.remove(product_cut) #Just in case there are two of the same species in the list, matching them by order.
                break
    return pairs
            

def join_map(rxn: 'ARCReaction',
             maps: List[List[int]]
             ) -> Optional[List[int]]:
    """
    a function that joins together the maps from the parts of the reaction.
    Args:
        rxn: ARCReaction that requires atom mapping
        maps: The list of all maps of the isomorphic cuts.

    Returns:
        an Atom Map of the compleate reaction.

    """
pass

# ROOH + ROOH <=> RO+ HOH +ROO
# RO OH ROO H <=> RO OH ROO H
# RO RO, OH OH, ROO ROO, H H
