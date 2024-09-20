# Hologram-Project
**'Diffraction-Informed Deep Learning for Molecular-Specific Holograms of Breast Cancer Cells '** \
Tzu-Hsi Song, Mengzhi Cao, Jouha Min, Hyungsoon Im, Hakho Lee, Kwonmoo Lee

## Architecture
<img src="/assets/Fig1.png" width=500>
Lens-free digital in-line holography (LDIH) offers a large field of view on the scale of 10 mm² at micrometer-scale resolution, capabilities not achievable with traditional lens-based microscopes. This makes LDIH a promising diagnostic tool for high-throughput cellular analysis. However, the complex nature of diffraction images (holograms) produced by LDIH poses challenges for human interpretation. To address these challenges, we present HoloNet, a novel deep learning architecture specifically designed for the direct analysis of diffraction images for cellular diagnosis. HoloNet is engineered to capture their inherent multi-scale features. It achieves superior performance by recognizing well-defined regions within complex diffraction images. Also, HoloNet effectively classifies breast cancer cell types and quantifies molecular marker intensities with high precision and interpretability. HoloNet offers a robust solution to the unique challenges of holographic data analysis, enhancing both the accuracy and interpretability of cellular diagnostics by seamlessly integrating computational imaging and deep learning.

## Installation
To install necessary library packages, run the following command in your terminal:
```
pip install -r requirements.txt
```

## Usage
* Clone the repo to your project folder by using the following commend:

    ``git clone https://github.com/kleelab-bch/HoloNet``


* Prepare the dataset as Excel file and copy to the ``Data`` folder. 
* Follow the order of codes (in the ``src`` folder)
  * Run ``1_Temporal_Clustering.py`` to obtain the cluster labels of US counties.
    * The Clustering labels will be saved to a new custom sheet in the original Excel file. 
  * Then run ``2_FIGINet_Prediction.py`` for model training and result forecasting.
    * If the user uses pretrained models , please set the parameter ``Use_Pretrained`` as True. 
* The forecasting results will be generated in ``Results`` folder 


## Note
- All the Covid-19 Confirmed Data of US Counties are from <a href="https://coronavirus.jhu.edu/">Center for Systems Science and Engineering (CSSE) at Johns Hopkins University</a>.
- The ``lib`` folder includes all dependencies required for the FIGInet workflow.
- All trained models are saved to the ``Model`` folder.

## Licence


## Contact
If you have any question about the date or code, please contact [kwonmoo.lee@childrens.harvard.edu](mailto:Kwonmoo.lee@childrens.harvard.edu)
