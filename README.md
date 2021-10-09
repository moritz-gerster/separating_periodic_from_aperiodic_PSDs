# Separating neural oscillations from aperiodic 1/f activity: challenges and recommendations 
#### Moritz Gerster, Gunnar Waterstraat, Vladimir Litvak, Klaus Lehnertz, Alfons Schnitzler, Esther Florin, Gabriel Curio, and Vadim Nikulin bioarXiv 2021 (Link coming soon)

- Computation_time.pynb: Code to compare computation time between FOOOF and IRASA
- FigX.pynb: Code to reproduce figure X from the article
- environment.yml: YAML file to create conda environment
- fooof_modified.py: [fooof](https://github.com/fooof-tools/fooof) functions modfied for visualization
- params.yml: Plot parameters for all figures
- requirements.txt: Conda requirements to run code
- utils.py: helper functions

Simulation data is created by the figure notebooks.
Patient data cannot be published due to patient data privacy restrictions.

## Abstract

Electrophysiological power spectra typically consist of two components: An aperiodic part usually following a 1/f power law P∝1/f^beta and periodic components appearing as spectral peaks. While the investigation of the periodic parts, commonly referred to as neural oscillations, has received considerable attention, the study of the aperiodic part has only recently gained more interest. The periodic part is usually quantified by center frequencies, powers, and bandwidths, while the aperiodic part is parameterized by the y-intercept and the 1/f exponent beta. For investigation of either part, however, it is essential to separate the two components.

In this article, we scrutinize two frequently used methods, FOOOF and IRASA, that are commonly used to separate the periodic from the aperiodic component. We evaluate these methods using diverse spectra obtained with electroencephalography (EEG), magnetoencephalography (MEG), and local field potential (LFP) recordings relating to three independent research groups. Each method and each dataset poses distinct challenges for the extraction of both spectral parts. The specific spectral features hindering the periodic and aperiodic separation are highlighted by simulations of power spectra emphasizing these features. Through comparison with the simulation parameters defined a priori, the parameterization error of each method is quantified. Based on the real and simulated power spectra, we evaluate the advantages of both methods, discuss common challenges, note which spectral features impede the separation, assess the computational costs, and propose recommendations on how to use them.

![\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}](https://latex.codecogs.com/svg.latex?\Large&space;x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}) 
