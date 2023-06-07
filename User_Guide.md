Four baseline models (DCA, scVAE, scDesign2, ZINB-WaVE) are runnable. ``mtx`` and ``h5ad`` files are supported. 
Details of running each model are listed below.

* **DCA**: Codes of running DCA can refer to the function ``normalAugmentExp`` in ``Models/dca/SplatDataExp.py``. 
Results will be saved into a npy file. The DCA model requires packages:   
``python==3.6, tensorflow==2.2, pyyaml==5.4, keras==2.4.3, pyopt==0.1.0, scanpy, torch==1.8.1``.

* **scDesign2**: Codes of running scDesign2 can refer to the function ``normalAugmentation`` in ``Models/scDesign2/SplatDataExp.R``. 
Results will be saved into a mtx file. The scDesign2 model requires packages:   
``scDesign2, Matrix``.  

* **scVAE**: Codes of running scVAE can refer to the function ``normalAugmentExp`` in ``Models/scvae/SplatDataExp.py``. 
Results will be saved into a npy file. The scVAE model requires packages:   
``python==3.6, tensorflow==1.15.2, torch==1.8.1, scanpy, imprtlib-resources, loompy, tensorflow-probability==0.7.0``. 
Notice that parameters of running scVAE should be listed in a JSON file, e.g. ``Models/scvae/splat_simulation_exp.json``. 

* **ZINB-WaVE**: Codes of running ZINB-WaVE can refer to the function ``normalAugmentation`` in ``Models/ZINBWaVE/SplatDataExp.R``. 
Results will be saved into a mtx file. The scVAE model requires packages:   
``zinbwave, Matrix, BiocParallel``. 