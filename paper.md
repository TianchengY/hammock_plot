---
title: 'Hammock_plot: A Python package for visualizing categorical data'
tags:
  - Python
  - plot
  - hammock plot
authors:
  - name: Tiancheng Yang
    orcid: 0009-0009-1009-8826
    equal-contrib: true
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
affiliations:
 - name: Tiancheng Yang, University of Waterloo, Canada
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 7 June 2024
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The hammock plot draws a graph to visualize categorical or mixed categorical / continuous data.
Variables are lined up parallel to the vertical axis. Categories within a variable are spread out along a
vertical line. Categories of adjacent variables are connected by boxes. (The boxes are parallelograms; we
use boxes for brevity). The "width" of a box is proportional to the number of observations that correspond
to that box (i.e. have the same values/categories for the two variables). The "width" of a box refers to the
distance between the longer set of parallel lines rather than the vertical distance.

If the boxes degenerate to a single line, and no labels or missing values are used the hammock plot
corresponds to a parallel coordinate plot. Boxes degenerate into a single line if barwidth is so small that
the boxes for categorical variables appear to be a single line. For continuous variables boxes will usually
appear to be a single line because each category typically only contains one observation.

The order of variables in varlist determines the order of variables in the graph.  All variables in varlist
must be numerical. String variables should be converted to numerical variables first, e.g. using encode or
destring.

TODO:add a summary of Python implementations

# Statement of need

TODO: "Neither Mosaic plots, scatter plot matrices nor parallel coordinate plots are well
suited for data that have both categorical and continuous variables. In Trellis displays one
specific plot (e.g. scatter plot or a box plot) is displayed for different subsets of
conditioning variables. These plots are then arranged as a panel. For example, one might
display two continuous and one categorical variable as a panel of scatter plots – one for
2
each category of the categorical variables. Therefore Trellis displays are suitable for
displaying mixed continuous / categorical data.
For survey researchers missing data are very important. Most plots do not to
accommodate missing data, presumably because the researchers who conceived these
plots did not work with surveys. Mosaic plots are an exception: missing values have
sometimes been added as an extra category. A similar approach is possible with scatter
plot matrices or parallel coordinate plots, but I have never seen this being done.
I introduce a new plot for the visualization of categorical data that also handles
interval data and mixed categorical /continuous data. I introduce the hammock plot in the
next section. The following section gives several examples. The paper concludes with a
brief discussion." Via Schonlau M. Visualizing Categorical Data Arising in the Health Sciences Using Hammock Plots. In Proceedings of the Section on Statistical Graphics, American Statistical Association; 2003

# Mathematics

TODO: Skip this part or add mathematics for calculating distance of parallelogram and rectangles

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References
