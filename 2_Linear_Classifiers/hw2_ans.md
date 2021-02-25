---
header-includes: |
     \usepackage{amssymb}
     \newcommand*{\vertbar}{\rule[-1ex]{0.5pt}{2.5ex}}
     \newcommand*{\horzbar}{\rule[.5ex]{2.5ex}{0.5pt}}
     \usepackage[vmargin={1in, 1in}, hmargin={1in, 1in}]{geometry}
urlcolor: blue     
---

# Homework 2 Answer

## Theory [1pt each]

1. Given a sample with feature vector $x=[1.1, 2.2, 3.3]^T$,  what is its augmented feature vector? 

$x=[1.1, 2.2, 3.3, 1]^T$

2. If the weight vector of a linear classifier is $w=[1, 0, 1, 0]^T$, and we define that a sample belongs to class $+1$ if $w^Tx>0$ and $-1$ if $w^Tx<0$ where $x$ is the augmented feature vector of the sample, what is the class of the sample? 

$w^Tx=4.4$, so this sample belongs to class +1

3. When discussing the sum of error squares loss function in the class, we used augmented but not normalized augmented (normalized and augmented) feature vectors. Please rewrite that loss function
$J(\mathbf{W}) = \sum_{i=1}^N (\mathbf{x}_i^T \mathbf{W} -y_i)^2$
 in terms of 
**normalized augmented** feature vectors. Let $x''_i$ be the normalized augmented feature vector of the $i$-th sample, and $w$ be the weight vector of the classifier. A correct prediction shall satisfy $w^Tx''_i>0$ regardless of the class of the sample because $x''_i$ has been normalized. You may use a computational algebra system to help -- but it is not required. It might be easier by hand on scratch paper.



4. Please find the solution for minimizing the new loss function. Keep variables and font style consistent with those in the class notes/slides, except that you can reuse the matrix
$\mathbb{X}=   \begin{pmatrix}
    \horzbar & \mathbf{x''}_1^T & \horzbar \\
    \horzbar & \mathbf{x''}_2^T & \horzbar \\
        &       \vdots        &   \\
    \horzbar & \mathbf{x''}_N^T & \horzbar \\
  \end{pmatrix}$, 
  each **row** of which is re-purposed into a normalized and augmented feature vector. The right most column of the new $\mathbb{X}$ should contain only $1$'s and $-1$'s.
