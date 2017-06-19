# Drug_Design_via_Support_Vector_Machine

A support vector machine-based algorithm for coarse-grained structure prediction of high-affinity ligands for a specific molecular target. Training data requires PDB structures of ligand-receptor complexes, mapped to the binding affinity of the interaction. The algorithm will attempt to generate the ratio of the distribution of atoms in each octant to the entire search space, given a query binding affinity of interaction.

Case 1: Was applied to the orphan G-protein-coupled receptor, SREB2, to probe potential structural features of its endogenous ligand/s. Coarse-grained ligand models, with varying specified binding affinities of interaction with SREB2, were generated. A Distinctive “Shifting” of ratios was found in certain regions, that was dependant on binding affinity (circled in red).

