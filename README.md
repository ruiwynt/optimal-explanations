# Optimal Minimal Explanations for Boosted Trees

## 1. Introduction

Given a boosted tree and data instance, computes the optimal explanation 

## 2. Background

### Universe of Classifiers 

A classification problem asks the question of how to label a given example of input data. Generally, a classifier has:

- A set of features $\mathcal{F} = \{1, 2, ..., m\}$. Each feature $j$ belongs to some domain $D_j$
- A set of classes $\mathcal{K} = \{c_1, c_2, ..., c_K\}$
- A feature space $\mathbb{F} = D_1 \times D_2 \times ... \times D_m$. A point in the feature space is an instance $\tilde{x} = (x_1, x_2, ..., x_m)$, with $x_j \in D_j$
- A classifier function $\tau: \mathbb{F} \rightarrow \mathcal{K}$, mapping each instance in the feature space to a class.

A tree ensemble is a set of decision trees which work together to compute an output label for a given instance. Each decision tree has an associated class, and there are $n$ decision trees for each class.Instances encode a path from the root to a leaf node. Decision trees have a set of non-terminal nodes which have a split condition on some feature, and leaf nodes have an associated weight such that when a path terminates at that leaf, its weight is added to the total weight of the class associated with that tree. The class with the highest total weight after computation over all trees is the output class.

More formally, let $\mathcal{E} = \{T_{Kj+i} : j \in [n], i \in [K]\}$ be an ensemble of decision trees, where $i$ is the class associated with tree $T_{Kj + i}$. Each decision tree is a set of non-terminal nodes $S$, terminal nodes $P$, and a function $T_{K_j + i}: \mathbb{F} \rightarrow \mathbb{R}$ which outputs the weight assigned by the tree for a given instance. Each non-terminal node has a split condition, which has an associated feature $j$, split value $s_{j,i}$, and relation operator in $(\leq, <, \geq, >)$. Let all split values across all trees for a particular feature be denoted $S_j = \{s_{j, 1}, s_{j, 2}, ..., s_{j, m_j}\}$. Note that the total class weight of an instance changes if and only if a feature value changes its ordering with respect to that feature's split values.

An **region** is a set of pairs $R = \{(l_j, u_j): j \in \mathcal{F}, l_j < u_j\}$. Each pair defines an interval for a particular feature. All of these pairs together define a subset of $\mathbb{F}$ where certain properties can be tested. For this universe, lets define the set of all regions $\mathcal{R}$ as all regions such that $l_j, u_j \in S_j$. 

Given a classifier $\tau(\tilde{x})$ and instance $\tilde{x}$ with output class $c$, a region $R$ is **explanative** if:

1. $l_j \leq x_j < u_j$ for all $j \in \mathcal{F}$
2. For any instance $\tilde{v}$ where $l_j \leq v_j \leq u_j$, $\tau(\tilde{v}) = c$. 

### Universe of SMT

This is the encoding of the classifier universe into SMT such that the optimal maximally explanative region (MaxER) with respect to some objective function $Z: \mathcal{R} \rightarrow \mathbb{R}$ can be found. The MARCO approach described in Liffiton et al. (2016) is implemented for subsets of regions and their threshold values.

We now describe how to encode any region into propositional logic. Recall that $S_j$ denotes the set of all split values for feature $j$. Let us sort $S_j$ such that $s_{j, a} < s_{j, b}$ if $a < b$. Then each pair of adjacent threshold values induces an **elementary interval** $I_{j, i} = [s_{j, i}, s_{j, i+1})$.

Let each elementary interval be denoted by a boolean variable $l_{j,i}$. Given a region $R$, let $l_{j,i}$ if and only if $l_j \leq s_{j, i}$ and $l_u \geq s_{j, i+1}$. Then each region $R$ can also be defined by a set of boolean variables $L$ such that $l_{j ,i} \in L$ if and only if $l_{j, i}$ is true.

Note that a region $A$ is contained within a region $B$ if and only if $L_A \subseteq L_B$.

Bottom of marco lattice is the empty set. Top of marco lattice is the set of all $l_{j,i}$.

### MaxSAT Entailment Checker

This section describes Algorithm 1 in Ignatiev et al. (2022). Given:

- An encoding of a boosted tree into hard clauses $\mathcal{H}$ and soft clauses $\mathcal{S}$.
- An input instance $\tilde{x}$, with its associated prediction $c_i$, in terms of boolean order variables as defined in Ignatiev et al. (2022).
- A candidate explanation $\chi$ in terms of the same boolean order variables, a subset of $\tilde{x}$.

The algorithm will output whether or not the candidate explanation entails the prediction $c_i$.

## 3. Procedure Outline

**OUTDATED, NEED TO UPDATE**

1. Take boolean variables which represent threshold values and correctly pair them up to create a *range variable*. For example, if $o_{ij} = 1$ iff $x_i < d_{ij}$ where $d_{ij}$ is the $j$-th largest threshold value for feature $i$, we can construct range variables $r_{ijk} = \lnot o_{ij} \land o_{ik}$ where $j < k$. We can retrieve the threshold values from the BT.

2. For each range variable $r_{ijk}$, calculate its weight $w_{ijk}$ as a monotone function of the interval size. For example, we can have that $w_{ijk} = \log(d_{ik} - d_{ij})$.

3. Take any instance of the feature space $\mathbf{x} = (x_1, x_2, ..., x_m)$. Discard all range variables which aren't consistent with the instance's feature values. For all range variables which are left, we can define the objective function to maximise as:

$$\sum_{\text{r not discarded}} w_{ijk}r_{ijk}$$

4. Construct the constraints for MARCO (Liffiton et al., 2016), which we use as a Partial Weighted MaxSAT solver. To do this, we feed the following constraints into the solver:
    -  The constraints on $o_ij$ from Ignatiev et al. (2022).
    -  A constraint mandating that only one $r_{ijk}$ is true for each feature $i$.
  
5. Run MARCO with the parameters defined in (3) and (4). It will return an optimal subset of range variables. This subset represents an assignment to variables $r_{ijk}$ which maximises the objective funciton subject to constraints.

6. Use the Entailment Check from Ignatiev et al. (2022) to check whether or not the optimal subset is a valid explanation for the given BT model. If it is a valid explanation, then we return the optimal explanation and we are done.

7. If the optimal subset is not a valid explanation, we construct a contrastive explanation (Ignatiev et al., 2020) from the optimal subset and feed it back into MARCO. This will block any subsets of the optimal subset from being explored by MARCO, reducing the search space.

8. Repeat steps 5 - 7 until an optimal valid subset is found.

## 4. References

Liffiton, M.H., Previti, A., Malik, A. et al. Fast, flexible MUS enumeration. Constraints 21, 223–250 (2016). https://doi.org/10.1007/s10601-015-9183-0

Ignatiev, A., Izza, Y., Stuckey, P. J., & Marques-Silva, J. (2022). Using MaxSAT for Efficient Explanations of Tree Ensembles. Proceedings of the AAAI Conference on Artificial Intelligence, 36(4), 3776-3785. https://doi.org/10.1609/aaai.v36i4.20292

Ignatiev, A., Narodytska, N., Asher, N., & Marques-Silva, J. (2020). On Relating 'Why?' and 'Why Not?' Explanations. ArXiv, abs/2012.11067.

## 5. Interesting things I came across

https://arxiv.org/pdf/2303.09271v1.pdf uses MARCO to calculate minimal explanations and minimum explanations. Claims to acheve a 2400x speedup compared to current SoTA methods.
