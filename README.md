# EVAX

EVAX stands for everyday argumentative explanations and is a model-agnostic, post-hoc method that computes argumentation framework (AF)-based explanations for decisions of ML classifiers. The explanations have contrastive, selected, and social characteristics; they include contrastive counterarguments, they consist of a fixed amount of arguments that can be selected based on a cognitive bias, and the size can be adjusted.


EVAX takes as input a labeled dataset, a trained black box model and a threshold value τ_select that controls the size of the output. EVAX returns a set of predictions and a set of local explanations The explanations answer the question: “Why did the black box assign class c to input instance x?” These explanations are deployments of an AF that represent the behavior of the black box around a single datapoint in argumentative terms. This AF thus forms the basis for the explanations, and will, for every classified instance, be referred to as the local AF. The size of this local AF can be manually altered by τselect. EVAX adopts a model-agnostic approach since it only uses the classifier as an oracle that can be queried for predictions.
