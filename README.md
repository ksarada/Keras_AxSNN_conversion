# Keras_AxSNN_conversion

These are the files required for deploying the parsed Keras model (.h5, model description and weights required), on the AxSNN C++ simulator, axsnnsim (cite link to Sanchari's codes)

Run Command : python WtRecord.py
(Note: Model path is to be provided as 'model_name' in WtRecord.py)

Output File containing the connection details, number of layers etc, ie, as expected by axsnnsim, will be dumped out as a single .txt file. No further changes need to be made. 

TODO (conversion):
1) Automate the identification of models that come with an input layer
2) Automate model specification
3) Stride specification, padding etc
4) Currently, only parsed models from snntoolbox(cite link) are supported as layers such as Activation, Dropout etc are removed. Must include this here, for all models.

TODO: (axsnnsim)
1) Max Pooling support
2) Single timestep specification of inputs

Note:
Bias support has been provided on axsnnsim by entering the bias value added to each neuron as the leak parameter
