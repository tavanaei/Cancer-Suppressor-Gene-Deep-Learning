# Cancer-Suppressor-Gene-Deep-Learning
</br>
</br>
## This project has been written in **Lua** and uses the **Torch** library.
</br>
.
</br>
To generate the Feature map sets:
</br>
  1. Create CSV files of protein tertiary structures as explained in the paper (see the liscence file). Three example-CSV files representing one protein have been uploaded to "SG" and "OG" folders.
  </br>
  2. Call the "Maper" function in Data.lua
  </br>
  3. Set up your preferred parameters by runing the Training.lua as follows:
  </br>
  .
  </br>
  
  $ *th Training.lua [Parameters]*
  </br>
  </br>
  Parameters:
  
  </br>
    **-positive**          Positive directory [OG_Map]
    </br>
    **-negative**          Negative Directory [SG_Map]
    </br>
    **-neutral**           Neutral Directory [UR_Map]
    </br>
    **-GPU**               preferred GPU [1]
    </br>
    **-nGPU**              No of GPUs [1]
    </br>
    **-kernel**            Kernels for convolution layers [16,32,32,64,64]
    </br>
    **-stride**            Stride values for Pooling [4,2,2,2]
    </br>
    **-hidden**            Hidden Layers [100,50]
    </br>
    **-iterations**        No of iterations [1]
    </br>
    **-batchSize**         Batch size [10]
    </br>
    **-learningRate**      Learning rate [0.01]
    </br>
    **-learningRateDecay** Learning rate decay [1e-05]
    </br>
    **-momentum**          Weight change history [0.6]
    </br>
    **-weightDecay**       regularizer parameter [0.0001]
    </br>
    **-cuda**              Use Cuda [false]
    </br>
    **-p**                 Kernel Size [7]
    </br>
    **-trainSize**         Training Samples [2029]
    </br>
    **-testSize**          Testing Samples [350]
    </br>
    **-validSize**         Validation Samples [0]
    </br>
    **-model**             Model File [Model.t7]
    </br>
    **-result**            Test Results of Target vs Predict [ResTest.dat]
    </br>
    .
    </br>

We only considered one GPU for this example. If you want to use more GPUs, please update the Training.lua by adding the DataParallelTable ...

Reference:


