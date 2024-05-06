# Digit Recognition

In the 'mnist' folder you have training data (images in .pbm format) with pre-trained model (data.nn). To train by yourself set ```TRAINING_BUILD``` in ```src/Config.h``` to 1, otherwise leave it 0.

Visualization is done using OpenGL with GLFW (see 'dependencies' folder).

Due to too few training data (around 1000 images) it is pretty sensitive, however usually it works.