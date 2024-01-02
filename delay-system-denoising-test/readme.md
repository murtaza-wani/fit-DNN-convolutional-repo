# Simulation Program for the Deep Learning Delay System (Fashion-MNIST denoising, validation on test set)

The following adaptations have to be done before compiling the code:

1. If you want to use a different activation function than f = sin and/or a different input preprocessing function than g = tanh, you need to modify the corresponding functions in f.h.

The program can be excecuted with the following option:

| flag                                   | default value     | explanation                                                                         |
| -------------------------------------- | ----------------- | ------------------------------------------------------------------------------------ |
| -filename [filename.txt]               | filename.txt      | Sets the filename of the textfile containing the simulation results.                  |
| -system_simu [name]                    | dde_heun          | Choose dde_heun for simulation with delay system.                                     |
| -grad_comp [name]                      | backprop_standard | Choose backprop_standard for dde_heun.                                                |
| -theta [decimal number]                | 0.5               | Node separation.                                                                     |                                                          | -number_of_epochs [integer number]     | 100               | Number of epochs.                                                                    |
| -eta0 [decimal number]                 | 0.001             | Learning rate eta = min(eta_0, eta_1 / step).                                        |
| -eta1 [decimal number]                 | 1000.0            |                                                                                      |










