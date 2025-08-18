<p align="left">
  <img src="docs/atems_logo.svg" alt= "# ATEMS" width="55%">
</p>

(***Py***thon ***A***nalysis tools for ***TEM*** images of ***S***oot)

------

A python codebase to analyze TEM images of soot, which includes new methods associated with this project and a compilation of other pre-existing methods into a single Python package. The original implementation was a port of previous MATLAB code (https://github.com/tsipkens/atems), which was described by [Sipkens et al. (2023)][joss1]. 





------

#### License

This software is released under a GNU GENERAL PUBLIC LICENSE license (see the corresponding license file for details).

#### Contributors and acknowledgements

This code was primarily compiled by Timothy A. Sipkens. 

This work was supported by Dr. Steven Rogak. Other direct contributors include [Hamed Nikookar](https://github.com/Hamed-NKR) and [Darwin Zhu](https://github.com/darwinz7). Pieces of this code were adapted from various sources and features snippets written by several individuals at UBC, including [Ramin Dastanpour](https://github.com/rdastanpour), [Una Trivanovic](https://github.com/unatriva), Alberto Baldelli, Yiling Kang, Yeshun (Samuel) Ma, and Steven Rogak, among others.

This program contains very significantly modified versions of the code distributed with [Dastanpour et al. (2016)][dastanpour2016]. The most recent version of the Dastanpour et al. code prior to this overhaul is available at https://github.com/unatriva/UBC-PCM (which itself presents a minor update from the original). That code forms the basis for some of the methods underlying the manual processing and the PCM method used in this code, as noted in the README above. However, significant optimizations have improved code legibility, performance, and maintainability (e.g., the code no longer uses global variables). 

A snapshot of the dm3_lib (https://github.com/nanobore/dm3) for reading in DM3 files is included in this program, and represents a Python adaptation of an original ImageJ plugin by Greg Jefferis. 

Also included with this program is an adaptation the MATLAB code of [Kook et al. (2015)][kook], modified to accommodate the expected inputs and outputs common to the other functions.

This code also contain an adaptation of **EDM-SBS** method of [Bescond et al. (2014)][bescond]. We thank the authors, in particular Jérôme Yon, for their help in understanding their original [Scilab code and ImageJ plugin](https://www.coria.fr/en/edm-sbs-automated-analysis-of-tem-images/). Modifications to allow the method to work directly on binary images (rather than a custom output from ImageJ) and to integrate the method into the MATLAB environment may present some minor compatibility issues, but allows use of the aggregate segmentation methods given in the **agg** package. 

The **carboseg** method follows from collaborative work with [Max Frei](https://github.com/maxfrei750) and is associated with [Sipkens et al. (2021)][ptech.cnn]. 

Finally, the progress bar in the function `tools.textbar`, which is used to indicate progress on some of the primary particle sizing techniques, is a modified version of that written by [Samuel Grauer](https://github.com/sgrauer). 

#### How to cite

On use of this code or the *k*-means segmentation procedure described [above](#k-means-segmentation-seg_kmeans), please cite: 

> [Sipkens, T. A., Rogak, S. N. (2021). Technical note: Using k-means to identify soot aggregates in transmission electron microscopy images. Journal of Aerosol Science, 105699.][jaskmeans]

Users of the pair correlation method (PCM), the Euclidean distance mapping-surface based scale (EBD-SBS), and Hough transform (following Kook et al.) codes should acknowledge the corresponding studies under the acknowledgements above. Please also consider citing this repository directly. 

When using the CNN method (e.g., carboseg) please cite

> [Sipkens, T.A., Frei, M., Baldelli, A., Kirchen, P., Kruis, F. E., & Rogak, S. N. (In Press). Characterizing soot in TEM images using a convolutional neural network. Powder Technology.][ptech.cnn]

and see the [CarbonBlackSegmentation](https://github.com/maxfrei750/CarbonBlackSegmentation) repository for information on training. 

#### References

[Bescond, A., Yon, J., Ouf, F. X., Ferry, D., Delhaye, D., Gaffié, D., Coppalle, A. & Rozé, C. (2014). Automated determination of aggregate primary particle size distribution by TEM image analysis: application to soot. Aerosol Science and Technology, **48**(8) 831-841.][bescond]

[Dastanpour, R., Boone, J. M., & Rogak, S. N. (2016). Automated primary particle sizing of nanoparticle aggregates by TEM image analysis. Powder Technology, **295** 218-224.][dastanpour2016]

[Dastanpour, R., & Rogak, S. N. (2014). Observations of a correlation between primary particle and aggregate size for soot particles. Aerosol Science and Technology, **48**(10) 1043-1049.][dastanpour2014]

[Kheirkhah, P., Baldelli, A., Kirchen, P. & Rogak, S., (2020). Development and validation of a multi-angle light scattering method for fast engine soot mass and size measurements. Aerosol Science and Technology, **54**(9), 1083-1101.][kheirkhah20]

[Kook, S., Zhang, R., Chan, Q. N., Aizawa, T., Kondo, K., Pickett, L. M., Cenker, E., Bruneaux, G., Andersson, O., Pagels, J., & Nordin, E. Z. (2016). Automated detection of primary particles from transmission electron microscope (TEM) images of soot aggregates in diesel engine environments. *SAE International Journal of Engines*, **9**(1) 279-296.][kook]

[Sipkens, T. A., Zhou., L., Rogak, S. N. (2020). Aggregate-level segmentation of soot TEM images by unsupervised machine learning. *European Aerosol Conference*, Aachen, Germany.][eac20]

[Sipkens, T. A., Rogak, S. N. (2021). Technical note: Using k-means to identify soot aggregates in transmission electron microscopy images. *Journal of Aerosol Science*, 105699.][jaskmeans]

[Sipkens et al., (2024). atems: Analysis tools for TEM images of carbonaceous particles. *Journal of Open Source Software*, **9**(99) 6416.][joss1]

[Sipkens, T.A., Frei, M., Baldelli, A., Kirchen, P., Kruis, F. E., & Rogak, S. N. (2021) Characterizing soot in TEM images using a convolutional neural network. Powder Technology.][ptech.cnn]

[Trivanovic, U., Sipkens, T. A., Kazemimanesh, M., Baldelli, A., Jefferson, A. M., Conrad, B. M., Johnson, M. R., Corbin, J. C., Olfert, J. S., & Rogak, S. N. (2020). Morphology and size of soot from gas flares as a function of fuel and water addition. *Fuel*, **279** 118478.][triv20]

[kook]: https://doi.org/10.4271/2015-01-1991
[dastanpour2016]: https://doi.org/10.1016/j.powtec.2016.03.027
[dastanpour2014]: https://doi.org/10.1080/02786826.2014.955565
[bescond]: https://doi.org/10.1080/02786826.2014.932896
[triv20]: https://doi.org/10.1016/j.fuel.2020.118478
[eac20]: https://doi.org/10.13140/RG.2.2.14433.12648
[jaskmeans]: https://doi.org/10.1016/j.jaerosci.2020.105699
[kheirkhah20]: https://doi.org/10.1080/02786826.2020.1758623
[ptech.cnn]: https://doi.org/10.1016/j.powtec.2021.04.026
[joss1]: https://doi.org/10.21105/joss.06416