# Optimal Minimal Explanations for Boosted Trees

## 1. Introduction

## 2. Procedure Outline

1. Take boolean variables which represent threshold values and correctly pair them up to create a *range variable*. For example, if $o_{ij} = 1$ iff $x_i < d_{ij}$ where $d_{ij}$ is the $j$-th largest threshold value for feature $i$, we can construct range variables $r_{ijk} = \lnot o_{ij} \land o_{ik}$ where $j < k$. We can retrieve the threshold values from the BT.

2. For each range variable $r_{ijk}$, calculate its weight $w_{ijk}$ as a monotone function of the interval size. For example, we can have that $w_{ijk} = \log(d_{ik} - d_{ij})$.

3. Take any instance of the feature space $\mathbf{x} = (x_1, x_2, ..., x_m)$. Discard all range variables which aren't consistent with the instance's feature values. For all range variables which are left, we can define the objective function to maximise as:

$$\sum_{\text{r not discarded}} w_{ijk}r_{ijk}$$

4. Construct the constraints which to be feed into MARCO (Liffiton et al., 2016), which we use as a Partial Weighted MaxSAT solver. To do this, we feed the following constraints into the solver:
  -  The constraints on $o_ij$ from Ignatiev et al. (2022).
  -  A constraint mandating that only one $r_{ijk}$ is true for each feature $i$.
  
5. Run MARCO to optimise the objective function in (3) subject to the constraints in (4). It will return an optimal subset of range variables $\chi = \{r_{ijk}\ |\ r_{ijk} = 1 \text{ iff objective function is maximised with constraints in the assignment}\}$.

6. Use the Entailment Check algorithm from Ignatiev et al. (2022) to check whether or not the optimal subset is a valid explanation for the given BT model. If it is a valid explanation, then we return the optimal explanation and we are done.

7. If the optimal subset is not a valid explanation, we construct a contrastive explanation (Ignatiev et al., 2020) from the optimal subset and feed it back into MARCO. This will block any subsets of the optimal subset from being explored by MARCO, reducing the search space.

8. Repeat steps 5 - 7 until an optimal valid subset is found.

## 4. References

Liffiton, M.H., Previti, A., Malik, A. et al. Fast, flexible MUS enumeration. Constraints 21, 223–250 (2016). https://doi.org/10.1007/s10601-015-9183-0

Ignatiev, A., Izza, Y., Stuckey, P. J., & Marques-Silva, J. (2022). Using MaxSAT for Efficient Explanations of Tree Ensembles. Proceedings of the AAAI Conference on Artificial Intelligence, 36(4), 3776-3785. https://doi.org/10.1609/aaai.v36i4.20292

Ignatiev, A., Narodytska, N., Asher, N., & Marques-Silva, J. (2020). On Relating 'Why?' and 'Why Not?' Explanations. ArXiv, abs/2012.11067.

## 5. Interesting things I came across

https://arxiv.org/pdf/2303.09271v1.pdf uses MARCO to calculate minimal explanations and minimum explanations. Claims to acheve a 2400x speedup compared to current SoTA methods.
