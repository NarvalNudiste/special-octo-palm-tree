[A cool website](http://playground.tensorflow.org)

# Various accuracy tests

## One layer, no dropout (relu)

### 1000 epochs : 

*Every tests have a batch size of 128*

8 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adadelta              | 65.104167%                  |
| sgd              | 63.541667%                  |
| adagrad              | 65.104167%                  |
| adamax              | 72.916667%                  |
| adam              | 78.645833%                  |


16 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adadelta              | 69.791667%                  |
| sgd              | 71.354167%                  |
| adagrad              | 68.229167%                  |
| adamax              | 69.270833%                  |
| adam              | 72.395833%                  |


32 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adadelta              | 71.875000%                  |
| sgd              | 63.541667%                  |
| adagrad              | 69.270833%                  |
| adamax              | 77.083333%                  |
| adam              | 72.916667%                  |


64 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adadelta              | 63.541667%                  |
| sgd              | 63.541667%                  |
| adagrad              | 66.666667%                  |
| adamax              | 73.437500%                  |
| adam              | 76.562500%                  |


### 10'000 epochs : 


8 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| sgd              | 66.666667%                  |
| adamax              | 36.458333%                  |
| adam              | 79.166667%                  |
| adadelta              | 63.541667%                  |
| adagrad              | 36.458333%                  |


16 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| sgd              | 63.541667%                  |
| adamax              | 75.000000%                  |
| adam              | 75.000000%                  |
| adadelta              | 76.041667%                  |
| adagrad              | 77.604167%                  |


32 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| sgd              | 63.541667%                  |
| adamax              | 75.520833%                  |
| adam              | 80.208333%                  |
| adadelta              | 36.458333%                  |
| adagrad              | 77.604167%                  |


64 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| sgd              | 63.541667%                  |
| adamax              | 73.437500%                  |
| adam              | 63.541667%                  |
| adadelta              | 73.958333%                  |
| adagrad              | 73.437500%                  |


## Two layer, no dropout (relu)

### 1000 epochs :

8 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adam              | 68.750000%                  |
| adagrad              | 72.916667%                  |
| sgd              | 63.020833%                  |
| adadelta              | 71.354167%                  |
| adamax              | 36.458333%                  |


16 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adam              | 76.562500%                  |
| adagrad              | 70.312500%                  |
| sgd              | 63.541667%                  |
| adadelta              | 71.354167%                  |
| adamax              | 71.354167%                  |


32 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adam              | 68.750000%                  |
| adagrad              | 68.229167%                  |
| sgd              | 36.458333%                  |
| adadelta              | 63.541667%                  |
| adamax              | 72.395833%                  |


64 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adam              | 63.541667%                  |
| adagrad              | 63.541667%                  |
| sgd              | 63.541667%                  |
| adadelta              | 69.791667%                  |
| adamax              | 71.875000%                  |


## Two layer, dropout (relu)

### 1000 epochs : 

8 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| sgd              | 63.020833%                  |
| adadelta              | 69.270833%                  |
| adagrad              | 66.145833%                  |
| adamax              | 72.395833%                  |
| adam              | 36.458333%                  |


16 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| sgd              | 64.062500%                  |
| adadelta              | 72.395833%                  |
| adagrad              | 69.270833%                  |
| adamax              | 73.437500%                  |
| adam              | 73.958333%                  |


32 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| sgd              | 67.187500%                  |
| adadelta              | 68.750000%                  |
| adagrad              | 36.458333%                  |
| adamax              | 63.541667%                  |
| adam              | 70.833333%                  |


64 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| sgd              | 63.541667%                  |
| adadelta              | 71.875000%                  |
| adagrad              | 63.541667%                  |
| adamax              | 72.916667%                  |
| adam              | 67.708333%                  |

#### Deeper networks : 

### 1000 epochs, 8 hidden layers

2 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adagrad              | 63.541667%                  |
| sgd              | 74.479167%                  |
| adamax              | 63.541667%                  |
| adam              | 63.541667%                  |
| adadelta              | 63.541667%                  |


4 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adagrad              | 70.312500%                  |
| sgd              | 65.625000%                  |
| adamax              | 55.208333%                  |
| adam              | 78.125000%                  |
| adadelta              | 73.958333%                  |


8 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adam              | 64.583333%                  |
| sgd              | 76.562500%                  |
| adagrad              | 73.437500%                  |
| adadelta              | 71.875000%                  |
| adamax              | 64.583333%                  |


16 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adam              | 71.875000%                  |
| sgd              | 70.312500%                  |
| adagrad              | 71.875000%                  |
| adadelta              | 69.791667%                  |
| adamax              | 73.437500%                  |


32 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adam              | 67.708333%                  |
| sgd              | 73.437500%                  |
| adagrad              | 69.791667%                  |
| adadelta              | 69.270833%                  |
| adamax              | 69.791667%                  |


64 Nodes (Dense) : 

| Optimizer         | Accuracy         |
|------------------|------------------:|
| adam              | 66.145833%                  |
| sgd              | 67.187500%                  |
| adagrad              | 66.666667%                  |
| adadelta              | 64.583333%                  |
| adamax              | 68.750000%                  |



