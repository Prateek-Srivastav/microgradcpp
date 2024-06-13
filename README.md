# micrograd.cpp
failed attempt to write micrograd_from_scratch from karpathy's video in cpp :')

there's some issue with backward function, as of now, you have to declare a separate ``Value`` variable for each operation, can't do like
```Value o = ((2*n).exp() - 1)/((2*n).exp() + 1)```
// which is formula for tanh non linearity

> it's another point that even doing this in elementary operations, doesn't provide the correct results.

also operator overloading for `dtype op Value` isn't providing with correct grads

mlp, layer and neuron classes are working, but due to the issue in grads, i couldn't use them to train a nueral network.

leaving this as it is for now, will check again later someday, hopefully :)

