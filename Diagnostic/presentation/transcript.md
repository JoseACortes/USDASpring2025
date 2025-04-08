*Hello, I am Jose Andres Cortes, this is my diagnostic presentation. My advisor is Dr Andrzej Korzeniowski. This research is done in collaboration with Drs Galina Yakubova, Aleksandr Kavetskiy, and Allen Tobert.*

# Outline
    - Background
    - Paper Review
    - Application of Paper Method
    - Conclusion

*Here I present a paper review on a spectrum deconvolution method and apply it to our current research*

# Background

## Application

*Soil nutrient measurements aid farming and emission management. 
Core harvesting is the common method to determine soil carbon content. Soil cores are taken to a lab for Dry Combustion, this gives precise measurements. Alternatively, the Mobile Inelastic Neutron Scattering System (MINS) Detects levels of carbon in a scanned region on site, avoiding the cost and time of lab analysis. 
This device has been in development by the USDA. As a research intern in mathematics I have been tasked with simulating then analyzing the results from the MINS system*

## Inelastic Neutron Scatterings

*Inelastic Neutron Scattering (INS) can occur when a neutron hits elements such as carbon. After collision, energy from the excited atom scatters out at characteristic energy signals. For example, a fast neutron colliding with the nucleus of a carbon atom at 14 MeV has a strong chance to yield a gamma ray at 4.44 MeV. The analysis of the MINS results relies on the measurement of these signals*

## Carbon Gamma Ray Cross Section

*This is the cross section of Gamma ray yields of carbon. Based on simulation of the neutron and carbon atom interaction. This shows the probability that gamma rays are produced at any specific energy level.*

## Simulating MINS - 1

*The Monte Carlo Neutron Particle Simulator, MCNP6.2. this package is developed by the nuclear team in los alamos, and is still in use today. It is the software package used to model the interactions that occur when neutrons are shot into the soil. Here we define the geometry and chemical makeup of all the objects in the system*

## Simulating MINS - 2

*The MINS system utilizes a neutron generator to emit fast neutrons which interact with carbon nuclei in the soil. All the events produce gamma radiation signatures that are captured by energy detectors.*

## Defining Spectrums

*The detector measures the energy pulses into a data format we call a spectrum. A spectrum is digitally defined with ordered bins, sometimes labeled with an energy level, at each bin there is a magnitude..
We call the results of one neutron being emitted a "History"
A single history has a time interval of 50 nanoseconds, starting when a neutron is emitted from the source. The energy deposited into the gamma detector during the history is tallied. Over sufficient histories, 10 to the power of 9 in this presentation, the spectrum is sufficiently resolved.
Furthermore, If normalized such that the sum of all values equals one, the spectrum represents the measured probability density function of the gamma detection for any history*

## Spectrum Analysis

*This is the spectrum resulting from simulating the MINS scanning a soil sample of an x% to y% carbon silicone mixture by weight, although the resulting spectrum comes from interactions with both elements, spikes of activity around characteristic energy levels are visible and prominent. We refer to these spikes as peaks, and they are the basis of analysis methods. The problem is formulated around taking these spectrums and deriving the elemental composition of the sample, we call this the deconvolution of spectra.*

## Data Generation

*For every trial I generate spectrums of 10 to the power of 9 histories, with the samples as varied mixtures from 0 to 30% carbon by weight, with silicone accounting for the remaining material.*

# Analysis Methods

*Analysis is conducted on all trials. 
I describe 5 analysis methods.
The classical way to do analysis is by measuring the size of the peak. This is done by finding a peak and baseline. *

## Classical Methods

*The sum of the baseline and peak should equal the spectrum. The baseline filters out all the data not caused by INS, with the remaining data measured as part of the characteristic peak. The area of the peak is correlated with the amount of carbon in the sample.
The Perpendicular Drop (PD) method sets the minimum of the element window as the baseline. 
This method is good for the prominent peaks found in lab analysis, but in the chaotic mobile setting on the field, baselines are not flat. 
In the Tangent Skim method, the baseline is the tangent line to the minimum values on both sides of the peak.*

## Peak Fitting - Linear Baseline

*In peak fitting, the parameterized sum of a peak function and baseline function are fit onto the measured spectrum. This is done using the Levenberg-Marquardt least squares method. The peak function is defined as a gaussian function with the shown parameters, and the baseline is linear*

## Regression - Levenberg-Marquardt method

*The Levenberg-Marquardt (L-M) algorithm is a widely used optimization technique for solving non-linear least squares problems. It combines gradient descent and the Gauss-Newton technique, making it effective for curve fitting and parameter estimation. The algorithm switches between the two techniques based on the convergence behavior using gradient descent for stability and Gauss-Newton for speed*

## Peak Fitting - Exponential Falloff

*since the underlying energy may not be linear, I have experimented with alternate baselines, including exponential falloff. As I continue to add more baselines, I've also had to develop parameter bounds, such as forcing the peak to stay within the minimum and maximum values of the target window*

## Prediction

*The peak areas are correlated to the carbon content of the soil, and the outermost values are used as the training data for linear regression in a final prediction, which implies this method is accurate within a fraction of a percentage.*

## Limitations

Peak and Baseline based methods are difficult to generalize to other elements, for every element you have to pick a new window, some elements have overlapping windows which mix characteristic signals. 

# Paper Review

*Published in 2019, Modeling of tagged neutron method for explosive detection using GEANT4 was published in the journal: Nuclear Instruments and Methods in Physics Research. The goal of this article is to identify explosive substances such as TNT or C4 by determining by their carbon (C), nitrogen (N), and oxygen (O) content to distinguish them from benign substances. This paper is one of many that was sent to us by our USDA counterparts*

## The Tagged Neutron Method

*The tagged neutron method refers to the overall method of measuring elements, rather than the analysis of spectra. 
architecturally this differs from MINS in that the detector is set to only record measurements that happen within a certain cone of coincidence*

## Component Curve Fitting - Training

*the method for analyzing the spectrum is a component "curve fitting". In this process, the spectrums from pure samples of elements are measured, this is the training data.*

## Component Fitting - Testing

*When testing containers for explosives, the tested data is assumed to be the result of interactions from a mix of the elements measured during training, and thus the tested spectrum would then be a mix of the training spectrums. A linear combination, with one parameter per element tested. This is fit by minimizing the cost function with the Levenberg-Marquardt (L-M) Algorithm.*

## Conclusion

*The method in their tests is accurate within 6% of a given element. Making it a very effective tool for identifying harmful substances in simulation*

# Application of Paper Method

*I apply this to my own simulated data, using the new architecture developed by the USDA team. I generated MCNP spectrums for 4 samples: Pure Carbon, Pure Silicon, al2o3 and a 50% Carbon 50% Silicon mix (by weight), setting the initial parameters as 1/n (n being the number of samples used in  testing)*

## Single Result

*First, fitting a linear combination of the pure carbon and silicone spectrums onto the mixture spectrum. This had the following result: also within 6% accurate*

## Ghost Element Limitation

*Investigating the inclusion of a incorrect element shows a limitation: although al2o3 was not present in the sample, when used as training data, the method gives a false positive of 14 percent*

## Results
*I applied the the component curve fitting method to the same simulated data as the other analysis methods. Also, using the boundaries as the training data*

## Full Results

*I compare the true carbon measurements of the tested spectrums with the predicted carbon measurements accross all the different methods*

## MAE Comparison

*When compared to the other methods for specifically carbon measurement, curve fitting balances between the accuracy of peak fitting methods and the simplicity of classical methods*

# Conclusion

*In summary, I explored a paper on spectral deconvolution and applied the curve-fitting method on the projects data. I compared this to the other methods. This method shows promise for real-time field analysis, offering a flexible alternative to peak-based approaches.*

# Future Work
*Looking forward, The simulations take a long time to run, about 5 days for 10 to the power of 9 histories. although access to the atlas hpc is available to run around 20 jobs at once, this is still a long time to wait for results, especially when testing thousands of samples as you would in a field. I will be looking into the accuracy of reversing the deconvolution method. By reversing the process, I can generate a spectrum from a known mixture of elements. This will allow me to test the accuracy of the deconvolution methods without having to wait for the long simulation times.*

## Acknowledgements

*Once again id like to thank my mentors at the math department and the USDA*

