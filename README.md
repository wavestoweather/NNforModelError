# NNforModelError
This is the code corresponding to Kriegmair et al, 2021

We use a concolutional neural network (CNN) to learn model error caused by unresolved scales. We the modified rotating shallow water (modRSW) model bu Kent etal, 2017 which includes highly nonlinearprocesses mimicking atmospheric convection. To create the training dataset we run the model in a high and a low resolution setup and compare the difference after one low resolution time step starting from the same initial conditions, thereby obtaining an exact target.

*Creating the training data*
- HR.py: Produces a the high resolution simulation (HR). 
  command line arguments: int s
- config_HR.py: Settings for HR.
- LR1.py: Coarse grains HR (LR1).
  command line arguments: int s 
- config_LR1.py: Settings for LR1.
- LR2.py: Propogates each time step of LR1 one time step forward with the low resolution model (LR2).
  command line arguments: int s 
- config_LR2.py: Settings for LR2.
- training_data.py: Generates the training data set from LR1 and LR2.
  command line arguments: int s
- config_training_data.py: Settings for the training data.
 
*Training the CNN*
- kerasConv1D.py: Trains the CNN
  command line arguments: int s
- config_kerasConv1D.py: Settings for the CNN.

*Verification*
- twinRun.py: Runs a low resolution simulation, with and without CNN correction.
  command line arguments: int s, int r, int e, string name, float weight
  
int s: seed number for the realization of the orography \
int r: The model time step of LR1 from which the forecasts are started \
int e: The epoch corresponding to the CNN \
string name: The name of the CNN \
float weight: the mass conservation constraint weighting
