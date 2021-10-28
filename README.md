[![DOI](https://img.shields.io/badge/Preprint-Gerster%20et%20al.%202021-brightgreen)](https://doi.org/10.1101/2021.10.15.464483)
[![DOI](https://img.shields.io/badge/python-3.9-blue)](https://doi.org/10.1063/5.0021420)
[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)

# Separating neural oscillations from aperiodic 1/f activity: challenges and recommendations 
**Moritz Gerster**, Gunnar Waterstraat, Vladimir Litvak, Klaus Lehnertz, Alfons Schnitzler, Esther Florin, Gabriel Curio, and Vadim Nikulin, bioRxiv 2021.10.15.464483; doi: https://doi.org/10.1101/2021.10.15.464483

#### Files:
- [Computation_time.ipynb](/Computation_time.ipynb): Code to compare computation time between FOOOF and IRASA
- FigX.pynb: Code to reproduce figure X from the article
- [environment.yml](environment.yml): YAML file to create conda environment
- [fooof_modified.py](fooof_modified.py): [fooof](https://github.com/fooof-tools/fooof) functions modfied for visualization
- [params.yml](params.yml): Plot parameters for all figures
- [requirements.txt](requirements.txt): Conda requirements to run code
- [utils.py](utils.py): helper functions

Simulation data is created by the figure notebooks.
Patient data cannot be published due to patient data privacy restrictions.

## Abstract

Electrophysiological power spectra typically consist of two components: An aperiodic part usually following a 1/f power law P &Proportional; 1/f<sup>&beta;</sup> and periodic components appearing as spectral peaks. While the investigation of the periodic parts, commonly referred to as neural oscillations, has received considerable attention, the study of the aperiodic part has only recently gained more interest. The periodic part is usually quantified by center frequencies, powers, and bandwidths, while the aperiodic part is parameterized by the y-intercept and the 1/f exponent &beta;. For investigation of either part, however, it is essential to separate the two components.

In this article, we scrutinize two frequently used methods, FOOOF and IRASA, that are commonly used to separate the periodic from the aperiodic component. We evaluate these methods using diverse spectra obtained with electroencephalography (EEG), magnetoencephalography (MEG), and local field potential (LFP) recordings relating to three independent research groups. Each method and each dataset poses distinct challenges for the extraction of both spectral parts. The specific spectral features hindering the periodic and aperiodic separation are highlighted by simulations of power spectra emphasizing these features. Through comparison with the simulation parameters defined a priori, the parameterization error of each method is quantified. Based on the real and simulated power spectra, we evaluate the advantages of both methods, discuss common challenges, note which spectral features impede the separation, assess the computational costs, and propose recommendations on how to use them. 

#### Figure 1
![Fig1](https://user-images.githubusercontent.com/45031224/136661949-bf33a4af-832f-450b-b9bc-d410729ee35f.png)
#### Figure 2
![Fig2](https://user-images.githubusercontent.com/45031224/136662000-c795386f-c54c-40d9-b89b-95509fa618fb.png)
#### Figure 3
![Fig3](https://user-images.githubusercontent.com/45031224/136662003-bf32fb77-a9b9-400a-b472-9daca911a0f0.png)
#### Figure 4
![Fig4](https://user-images.githubusercontent.com/45031224/136662007-a1ef4cad-b90f-4519-be02-511168f0d9d7.png)
#### Figure 5
![Fig5](https://user-images.githubusercontent.com/45031224/136662010-dd9e46b5-88e5-4d81-848c-ade800d05bf5.png)
#### Figure 6
![Fig6](https://user-images.githubusercontent.com/45031224/136662011-8d915f5c-ce1a-45c2-9bd2-655936c68a17.png)
#### Figure 7
![Fig7](https://user-images.githubusercontent.com/45031224/136662016-b0a1f5d7-c603-44bb-9f78-24c3face961f.png)
#### Figure 8
![Fig8](https://user-images.githubusercontent.com/45031224/136662019-6fb6557e-6b00-49cd-9b06-2037d762a003.png)
