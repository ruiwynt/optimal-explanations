# Optimal Minimal Explanations for Boosted Trees

## Procedure Outline
*WIP*

1. Take boolean variables which represent threshold values and correctly pair them up to create a *range variable*. For example, if $o_{ij} = 1$ iff $x_i < d_{ij}$ where $d_ij$ is the $j$-th largest threshold value for feature $i$, we can construct range variables $r_{ijk} = \lnot o_{ij} \land o_{ik}$ where $j < k$. We can retrieve the threshold values from the BT.

2. For each range variable $r_{ijk}$, calculate its weight $w_{ijk}$ as a monotone function of the interval size. For example, we can have that $w_{ijk} = \log(d_ik - d_ij)$.

3. Take any instance of the feature space $\mathbb{x} = {x_1, x_2, ..., x_m}$. Discard all range variables which aren't consistent with the instance's feature values. For all range variables which are left, we can define the objective function to maximise as:

$$\sum_{\text{ijk not discarded}} w_{ijk}r_{ijk}$$

4. Construct the constraints which we will fit into MARCO (which we treat as a Partial Weighted MaxSAT solver). To do this, we take:
-  The constraints on $o_ij$ from Ignatiev et al. (2022).
-  A constraint mandating that only one $r_{ijk}$ is true for each feature $i$.

## References

Ignatiev, A., Izza, Y., Stuckey, P. J., & Marques-Silva, J. (2022). Using MaxSAT for Efficient Explanations of Tree Ensembles. Proceedings of the AAAI Conference on Artificial Intelligence, 36(4), 3776-3785. https://doi.org/10.1609/aaai.v36i4.20292
