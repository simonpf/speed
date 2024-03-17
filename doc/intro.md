# The IPWG Satellite Precipitation Estimation Evaluation Dataset (SPEED)

## Introduction

The Satellite Precipitation Estimation Evaluation Dataset (SPEED) has been
developed by the International Precipitation Working Group (IPWG) to serve as a
benchmark and validation dataset for precipitation retrievals. The dataset
consists of collocations of various space-borne passive microwave sensors with
reference precipitation measurements from ground-based and space-borne active
sensors.

The dataset aims to serve two principal applications:

 1. The rapid benchmarking of precipitation retrieval algorithms
 2. The validation of operational precipitation retrieval algorithms

SPEED consists of collocations of satellite *observations* from passive
microwave and IR sensors and corresponding *reference data* in the form of
precipitation estimates from ground-based and space-borne precipitation radars.
Collocations are provided both on the native grid of the *observations* and
regridded to a global, regular latitude-longitude with a resolution of
0.036$^\circ$.

### Use cases

SPEED supports two principal use cases, which we refer to as *validation* and
*benchmarking*. Although these two use cases are related, they differ slightly with respect to their aims and requirements.

#### Validation

The aim of retrieval validation is to quantify the absolute retrieval
uncertainty in precipitation estimates of a retrieval system in its entirety.
For validation, the retrieval system is evaluated in an end-to-end manner and
the resulting uncertainties thus include the effects of all components of the
retrieval system including the retrieval algorithm, implicit or explicit a
priori assumptions, and the processing system.

Although validation generally aims to quantify uncertainties in absolute terms,
in practice, this is typically not possible due to uncertainties in the
reference. However, in particular for data-driven retrievals, it is important to
evaluate the estimates using independent observations and reference data to
ensure the resulting validation result includes the generalization error.

#### Benchmarking

The aim of benchmarking is the comparison of retrieval algorithms. Since the
retrieval algorithm constitutes a principal component of a retrieval system,
retrieval development typically devotes a significant effort to optimizing this
component. In particular with respect to the recent consolidation of
deep-learning-based precipitation retrievals and the diversity of available
implementations, there is a need for ranking the accuracy of different
algorithms.

*Benchmarking* aims to assess retrieval algorithms alone thus excluding the
effects of a priori assumptions and other components of the retrieval systems
that may have confounding effects on the retrieval accuracy. In order to aid the development of retrieval systems, benchmarking should have reduced computational requirements to support rapid evaluation of retrieval algorithms. 


#### Implementation

Figure 1 illustrates the suggested processing flows for the validation and
benchmarking use cases. The SPEED collocations are at the base of both use
cases. In addition to the collocation dataset, SPEED defines an evaluation
framework for both use cases. This framework is implemented by the ``speed``
Python pacakge. The three principal functionalities (denoted by 1-3 in Figure 1)
provided by the pacakge are:
    
 1. Extraction of training and testing datasets
 2. An interface for running generic retrieval algorithms on the SPEED collocation dataset
 3. Calculation of accuracy metrics

#### Design rationale

SPEED aims to advance global precipitation remote sensing by providing a unified
evaluation framework for precipitation retrievals. An important component of
this is the rapid benchmarking of retrieval algorithms, in particular
machine-learning-based retrievals, to further the understanding of retrieval
methodology and support the rapid evaluation and adoption of advanced machine-learning techniques.

However, benchmarking results in the form of accuracy metrics on test data derived from the same source as the training data only provide a upper bound on the retrieval uncertainty and don't necessarily provide a suitable estimate of the accuracy of the resulting retrieval system. To ensure novel retrieval techniques are developed to improve global precipitation estimates in an effective manner, SPEED also defines a framework to validate precipitation estimates.

To support these two use cases,  SPEED comprises a dataset of core
collocations. Training and testing datasets can be derived from this dataset in
a deterministic manner ensuring comparibility of algorithms trained on these
datasets. Since different algorithms have different requirements regarding the
structure of the training dataset, this approach ensures that SPEED supports a
wide variety of retrieval algorithms.

The core collocations form the basis for the validation of retrieval algorithms.
The collocations define a unique distribution of input observations that is used
to evaluate precipitation retrievals. This is required because the training and
testing datasets derived for different retrieval methods may lead to different
observation sampling due to, for example, differences in the dataset structure between image-based and pixel-based retrievals.

Finally, the extraction of the core collocation dataset is the conceptually and
computationally most demanding task in the dataset generation for both use cases
supported by SPEED. By supporting both validation and benchmarking, SPEED aims
to increase the uesfulness of the resulting dataset. Finally, the collocation
dataset can potentially support additional applications such as comparing
capabilities of different sensors and the benchmarking of radiative transfer
simulations.
