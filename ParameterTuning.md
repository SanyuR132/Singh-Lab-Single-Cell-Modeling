## Parameter Tuning 

**Task**: Tune parameters of four baselines (DCA, scVAE, scDesign2, and ZINBWaVE) on four datasets
 (new mouse cell, WOT, and Zebrafish).

In [`./(REFERENCE)AutoTune.py`](./(REFERENCE)AutoTune.py), I implemented a runnable tuning
 framework for DCA and scVAE based on `optuna` package .
But they were not tested carefully and more parameters can be tuned. This is just a guide
 framework and feel free to revise it when necessary. The other two models (scDesign2 and
  ZINBWaVE) are R-based, I recommend using [ParBayesianOptimization](https://github.com/AnotherSamWilson/ParBayesianOptimization)
for parameter tuning.



### Models

- **DCA**: Documentations of DCA may refer to https://github.com/theislab/dca. 
Example codes of running DCA is the function ``normalAugmentExp`` in [``Models/dca/MouseDataExp.py``](./dca/MouseDataExp.py). The DCA model requires packages:   
``python==3.6, tensorflow==2.2, pyyaml==5.4, keras==2.4.3, pyopt==0.1.0, scanpy, torch==1.8.1``.


- **scVAE**: Documentations of scVAE may refer to https://scvae.readthedocs.io/en/latest/. 
Example codes of running scVAE is the function ``normalAugmentExp`` in [``./scvae/MouseDataExp.py``](./scvae/MouseDataExp.py). The scVAE model requires packages:   
``python==3.6, tensorflow==1.15.2, torch==1.8.1, scanpy, imprtlib-resources, loompy, tensorflow-probability==0.7.0``. 
Notice that parameters of running scVAE should be listed in a JSON file, e.g. [``./scvae/splat_simulation_exp.json``](./scvae/splat_simulation_exp.json).


* **scDesign2**: Documentations of DCA may refer to https://github.com/JSB-UCLA/scDesign2. 
Codes of running scDesign2 can refer to the function ``normalAugmentation`` in [``./scDesign2/MouseDataExp.R``](./scDesign2/MouseDataExp.R). 
The scDesign2 model requires packages:   ``scDesign2, Matrix``.  


* **ZINB-WaVE**: Documentations of ZINB-WaVE may refer to https://bioconductor.org/packages/devel/bioc/vignettes/zinbwave/inst/doc/intro.html. 
Codes of running ZINB-WaVE can refer to the function ``normalAugmentation`` in [``./ZINBWaVE/MouseDataExp.R``](./ZINBWaVE/MouseDataExp.R). 
The scVAE model requires packages: ``zinbwave, Matrix, BiocParallel``. 


### Datasets

Data are avalable at https://drive.google.com/file/d/1hY1fv-GTV2G7j7u6OKuO0_rqOkfVcsli/view?usp=sharing
