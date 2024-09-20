# Hologram-Project
**'Diffraction-Informed Deep Learning for Molecular-Specific Holograms of Breast Cancer Cells '** \
Tzu-Hsi Song, Mengzhi Cao, Jouha Min, Hyungsoon Im, Hakho Lee, Kwonmoo Lee

## Architecture
<img src="/assets/Fig1.png" width=500>
Lens-free digital in-line holography (LDIH) offers a large field of view on the scale of 10 mm² at micrometer-scale resolution, capabilities not achievable with traditional lens-based microscopes. This makes LDIH a promising diagnostic tool for high-throughput cellular analysis. However, the complex nature of diffraction images (holograms) produced by LDIH poses challenges for human interpretation. To address these challenges, we present HoloNet, a novel deep learning architecture specifically designed for the direct analysis of diffraction images for cellular diagnosis. It effectively classifies breast cancer cell types and quantifies molecular marker intensities with high precision and interpretability. HoloNet offers a robust solution to the unique challenges of holographic data analysis, enhancing both the accuracy and interpretability of cellular diagnostics by seamlessly integrating computational imaging and deep learning.

## Installation
To install necessary library packages, run the following command in your terminal:
```
pip install -r requirements.txt
```

## Instructions
* Clone the repo to your project folder by using the following commend:

    ``git clone https://github.com/kleelab-bch/HoloNet``

* Prepare the dataset as mat file and copy to the ``Data`` folder. 
* Follow the order of codes
  * Run ``main.py`` to get the model and prediction results.
    * Please read the instructions in ``main.py`` to swtich different models. 
* The results will be printed on the terminal. 

## Note
- The data is saved as mat format. If the user wnats to change it, the data collection function is in the Utilities.py  
- The ``lib`` folder includes all dependencies required for the HoloNet and related models.
- All trained models are saved to the ``Model_Save`` folder.

## License
This project is licensed under the [MIT License](LICENSE).

## Contact
If you have any question about the date or code, please contact [kwonmoo.lee@childrens.harvard.edu](mailto:Kwonmoo.lee@childrens.harvard.edu)
