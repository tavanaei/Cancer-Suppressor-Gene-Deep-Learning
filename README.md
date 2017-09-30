# Cancer Suppressor Gene and Oncogene Prediction using Deep Learning (CNN)

## This project has been written in **Lua** and uses the **Torch** library.

#### To generate the Feature map sets:
  1. Create CSV files of protein tertiary structures as explained in the paper (see the license file). Three example-CSV files representing one protein have been uploaded to "SG" and "OG" folders.
  2. Call the "Maper" function in Data.lua
  3. Set up your preferred parameters by runing the Training.lua as follows:
  
  > $ *th Training.lua [Parameters]*
  
  ### Parameters:
  
  
    -positive         Positive directory [OG_Map]
    
    -negative          Negative Directory [SG_Map]
 
    -neutral           Neutral Directory [UR_Map]
    
    -GPU               preferred GPU [1]
   
    -nGPU              No of GPUs [1]
   
    -kernel            Kernels for convolution layers [16,32,32,64,64]
   
    -stride            Stride values for Pooling [4,2,2,2]
   
    -hidden            Hidden Layers [100,50]
   
    -iterations        No of iterations [1]
    
    -batchSize         Batch size [10]
    
    -learningRate      Learning rate [0.01]
    
    -learningRateDecay Learning rate decay [1e-05]
   
    -momentum          Weight change history [0.6]
   
    -weightDecay       regularizer parameter [0.0001]
    
    -cuda              Use Cuda [false]
    
    -p                 Kernel Size [7]
   
    -trainSize         Training Samples [2029]
   
    -testSize          Testing Samples [350]
    
    -validSize         Validation Samples [0]
   
    -model             Model File [Model.t7]
    
    -result            Test Results of Target vs Predict [ResTest.dat]
    

##### We only considered one GPU for this example. If you want to use more GPUs, please update the Training.lua by adding the DataParallelTable ...

##### Reference:

Tavanaei, Amirhossein, Anandanadarajah Nishanth, Anthony Maida, and Rasiah Loganantharaj, "*A Deep Learning Model for Predicting Tumor Suppressor Genes and Oncogenes from PDB Structure*", 
doi: 10.1101/177378, bioRxiv, 2017.



