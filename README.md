# EVAX

EVAX is an argumentative explanation method. It belongs to the AI master thesis by Jowan van Lente, Utrecht University (2022).


EVAX stands for everyday argumentative explanations and is a model-agnostic, post-hoc method that computes argumentation framework (AF)-based explanations for decisions of ML classifiers. The explanations have contrastive, selected, and social characteristics; they include contrastive counterarguments, they consist of a fixed amount of arguments that can be selected based on a cognitive bias, and the size can be adjusted.


EVAX takes as input a labeled dataset, a trained black box model and a threshold value τ_select that controls the size of the output. EVAX returns a set of predictions and a set of local explanations The explanations answer the question: “Why did the black box assign class c to input instance x?” These explanations are deployments of an AF that represent the behavior of the black box around a single datapoint in argumentative terms. This AF thus forms the basis for the explanations, and will, for every classified instance, be referred to as the local AF. The size of this local AF can be manually altered by τselect. EVAX adopts a model-agnostic approach since it only uses the classifier as an oracle that can be queried for predictions.

The code uses datasets from the UCI machine learning repository (https://archive.ics.uci.edu/ml/index.php) and machine learning models from the scikit-learn library (https://scikit-learn.org/stable/). To test EVAX on the same datasets these should first be downloaded.



Information about thesis:

Title: Everyday argumentative explanations for AI

Abstract:
There has been an upswing in the research field of explainable artificial intelligence (XAI) of methods aimed at explaining opaque artificial intelligence (AI) systems and their decisions. A recent, promising approach involves the use of formal argumentation to explain machine learning (ML) applications. In this thesis we investigate that approach; we aim to gain understanding of the value of argumentation for XAI. In particular, we explore how well argumentation can produce everyday explanations. Everyday explanations describe how humans explain in day-to-day life and are claimed to be important for explaining decisions of AI systems to end-users. First, we conceptually show how argumentative explanations can beposed as everyday explanations. Afterward, we demonstrate that current argumentative explanation methods compute explanations that already contain some, but not all properties ofeveryday explanations. Finally, we present everyday argumentative explanations, or EVAX, which is a model-agnostic method that computes local explanations for ML models. These explanations can be adjusted in their size and retain high fidelity scores (an average of 0.95) on four different datasets and four different ML models. In addition, the explanations incorporate the main characteristics of everyday explanations and help in achieving the objectives of XAI.


