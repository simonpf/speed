# Overview

SPEED consists of collocations of satellite *observations* from passive
microwave and IR sensors and corresponding *reference data* in the form of
precipitation estimates from ground-based and space-borne precipitation radars.
Collocations are provided both on the native grid of the *observations* and
regridded to a global, regular latitude-longitude grid with a resolution of
0.036$^\circ$.

## Use cases

SPEED supports two principal use cases, denoted as *validation* and
*benchmarking*. Although these two use cases are related, they differ slightly
with respect to their aims and requirements.

### Validation

The aim of retrieval validation is to quantify the retrieval uncertainty in
precipitation estimates of a retrieval system in its entirety. For validation,
the retrieval system is evaluated in an end-to-end manner and the resulting
uncertainties thus include the effects of all components of the retrieval system
including the retrieval algorithm, implicit or explicit a priori assumptions,
and the processing system.

Although validation generally aims to quantify uncertainties in absolute terms,
this is typically not possible in practice due to uncertainties in the reference
data itself. However, in particular for data-driven retrievals it is important
to evaluate the resulting estimates using independent reference data to ensure
that the validation results take into account the error caused by uncertainties
in the reference data that was used in the development of the data-driven
retrieval.

### Benchmarking

The aim of benchmarking is the comparison of retrieval algorithms in isolation
from the other components of the retrieval system. Since the retrieval algorithm
constitutes a principal component of every retrieval systems, retrieval
development typically devotes a significant effort to optimizing this component.
In particular with due to consolidation of deep-learning-based retrieval
algorithms and the diversity of available architecture, there is a need for
a ranking the accuracy of different algorithms.

*Benchmarking* aims to assess retrieval algorithms alone thus excluding the
effects of a priori assumptions and other components of the retrieval systems
that may have confounding effects on the retrieval accuracy. In order to aid the
development of retrieval systems, benchmarking should have reduced computational
requirements to support rapid evaluation of retrieval algorithms.


## Implementation

Figure 1 illustrates the suggested processing flows for the validation and
benchmarking use cases. The SPEED collocations are at the base of both use
cases. In addition to the collocation dataset, SPEED defines an evaluation
framework for both use cases. This framework is implemented by the ``speed``
Python pacakge. The three principal functionalities (denoted 1-3 in Figure 1)
provided by the pacakge are:
    
 1. Extraction of training and testing datasets
 2. An interface for running generic retrieval algorithms on the SPEED collocation dataset
 3. Calculation of accuracy metrics

```{figure} images/use_cases.png
---
width: 75%
name: use_cases
---
SPEED use cases. The diagram illustrates the two principal uses cases of SPEED, which are precipitation retrieval validation and retrieval algorithm benchmarking. The core component of SPEED is a dataset of collocations. For the validation use case, SPEED provides a generic Python interface to run user-provided retrieval algorithms on the colocations. For the evaluation of the resulting retrieval results, SPEEe provides functionality to derive a pre-defined selection of accuracy metrics. For the benchmarking of retrieval algorithms, SPEED provides functionality to extract training and testing datasets in a deterministic manner. The algorithms are evaluated using the metrics defined by SPEED. 
```

## Collocation examples

Figures {numref}`example_combined` and {numref}`example_mrms` display two
collocations of AMSR2 observations with reference data from GPM CMB and MRMS,
respectively.

```{figure} images/collocation_example_amsr2_combined.png
---
width: 75%
name: example_combined
---
Example collocation of AMSR2 observations and GPM CMB reference data. The left-most column of panels shows the observations from the 10.62 GHz channels of the AMSR2 sensor on the GCOM-W1 platform. The center column shows collocated geostationary IR brightness temperatures. The right-most column shows the surface precipitation estimates from the GPM CMB product. The first row of panel shows the collocations in mapped to the native grid of the AMSR2 observations. The second row of panels show the same observations remapped to the regular latitude longitude grid.
```

```{figure} images/collocation_example_amsr2_mrms.png
---
width: 75%
name: example_mrms
---
Same as Fig. {numref}`example_combined` but for collocations with MRMS measurements.
```

## Design rationale

The purpose of SPEED is to advance global precipitation remote sensing by
providing a unified evaluation framework for precipitation retrievals. An
important component of this is the  benchmarking of retrieval algorithms,
in particular machine-learning-based retrievals, to further the understanding of
retrieval methodology and support the rapid evaluation and adoption of advanced
machine-learning techniques.

However, simple benchmarking results, i.e. retrieval accuracy evaluated on test
data derived from the same source as the training data, can only provide an
upper bound on the absolute retrieval uncertainties of the retrieval system.
Prior to operational application of a novel algorithm, the complete system must
be validated to ensure the improved *benchmark results* also lead to absolute
improvements in global precipitation estimates. To accelerate the adoption of
these advanced retrieval algorithms for operational use, SPEED also defines a
framework to validate the resulting retrieval systems.

To support these two use cases, SPEED consists of a dataset of core
collocations. Training and testing datasets can be derived from this dataset
using the accompanying software package in a deterministic manner, thus ensuring
comparibility of algorithms trained on these datasets. Since different
algorithms have different requirements regarding the structure of the training
dataset, this approach also ensures that SPEED supports a wide variety of
retrieval algorithms.

The core collocations form the basis for the validation use case. The
collocations uniquely define distribution of input observations that is used to
evaluate precipitation retrievals. Since the training and testing datasets
derived for different retrieval methods may deviate from the temporospatial
sampling of the core collocations, for example due the differences in the
dataset structure between image-based and pixel-based retrievals, such an
independent collocation dataset is required to ensure fair comparison between
different retrievals. Since the core collocations  retain all the data
available in the L1C data, they also support the validation of user-provided
retrieval algorithms that were developed independently of the benchmarking
datasets. The collocations thus not only allow for the benchmarking of
machine-learning-based retrievals but also comparing them against traditional
retrieval methods.

Finally, the extraction of the core collocation dataset is the conceptually and
computationally most demanding task in the dataset generation for both use cases
supported by SPEED. By supporting both validation and benchmarking, SPEED aims
to increase return on the effort required to create the collocation dataset.
Moreover, by making the collocation dataset publicly available SPEED can support
additional applications that benefit from collocations of PMW observations and
reference precipitation estimate, such as the evaluation of difference sensor
type or the benchmarking of radiative transfer simulations.

