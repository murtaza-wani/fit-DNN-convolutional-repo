# Simulation Program for the Deep Learning Delay System

The following adaptations have to be done before compiling the code:

1. If you want to use a different activation function than f = sin and/or a different input preprocessing function than g = tanh, you need to modify the corresponding functions in f.h.

2. If you want to use a task other than MNIST or Fashion-MNIST, you need to uncomment the corresponding lines in the file global_constants.h.

The program can be excecuted with the following option:

| flag                                   | default value     | explanation                                                                                                                                        |
| -------------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- 
   |
| -filename [filename.txt]               | filename.txt      | Sets the filename of the textfile containing the simulation results.                                                                               |
| -task [name]                           | MNIST             | MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100-coarse and SVHN are available options.                                                                   |
| -system_simu [name]                    | dde_heun          | Choose dde_heun for simulation with delay system
   |                                      
| -grad_comp [name]                      | backprop_standard | Choose backprop_standard for dde_ibp
   |
| -theta [decimal number]                | 0.5               | Node separation.                                                                                                                                   |                                                                                                                                          
| -number_of_epochs [integer number]     | 100               | Number of epochs.                                                                                                                                  |
| -eta0 [decimal number]                 | 0.001             | learning rate eta = min(eta_0, eta_1 / step)                                                                                                       |
| -eta1 [decimal number]                 | 1000.0            |                                                                                                                                                    |
| -pixel_shift                           | false             | Enable random pixel shift for data augmentation.                                                                                                   |
| -max_pixel_shift [integer number]      | 1                 | Maximum distance for pixel shift.                                                                                                                  |
| -training_noise                        | false             | Enable noise for data augmentation.                                                                                                                |
| -max_pixel_shift [integer number]      | 0.01              | Standard deviation of Gaussian training noise.                                                                                                     |
| -rotation                              | false             | Enable random rotation for data augmentation. Use only for CIFAR-10.                                                                               |
| -max_rotation_degrees [decimal number] | 15.0              | Maximum degree for rotation.                                                                                                                       |
| -flip                                  | false             | Enable random horizontal flip for data augmentation. Use only for CIFAR-10.                                                                        |
