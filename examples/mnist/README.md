This package contains a short example of how to use `go-deeper` for training 
and evaluating model performance on the MNIST dataset.

The following steps show how to clone and run the example:

1. `git clone https://github.com/white43/go-deeper.git`
2. `cd go-deeper/examples/mnist`
3. `./download.sh`
4. `go run ./...`

They should result in something like the snippet below:

```
Loaded 60000 images in 10 classes for training and 10000 for validation

Epoch 1 (7.94 sec), loss: 1.9426, val_acc: 0.5706, lr: 0.1000
Epoch 2 (8.34 sec), loss: 1.8430, val_acc: 0.7061, lr: 0.1000
Epoch 3 (8.45 sec), loss: 1.8583, val_acc: 0.7573, lr: 0.1000
Epoch 4 (8.56 sec), loss: 1.8385, val_acc: 0.7882, lr: 0.1000
Epoch 5 (8.51 sec), loss: 1.6206, val_acc: 0.8057, lr: 0.1000
Epoch 6 (8.62 sec), loss: 1.6706, val_acc: 0.8213, lr: 0.1000
Epoch 7 (8.69 sec), loss: 1.7028, val_acc: 0.8329, lr: 0.1000
Epoch 8 (8.55 sec), loss: 1.7342, val_acc: 0.8437, lr: 0.1000
Epoch 9 (8.79 sec), loss: 1.7184, val_acc: 0.8502, lr: 0.1000
Epoch 10 (8.48 sec), loss: 1.7324, val_acc: 0.8576, lr: 0.1000

Training Accuracy: 0.855567

              0     1     2     3     4     5     6     7     8     9  Recall
        0  5497     0    60    51    18   119    72    21    71    14   92.81 
        1     1  6462    36    38     7    67    13    25    79    14   95.85 
        2    84    45  4863   188   151    41   217   141   196    32   81.62 
        3    54    37   233  4966    23   348    40   106   230    94   81.00 
        4    35    32    63    23  5013    69   110    48    61   388   85.81 
        5   132    45    75   339    97  4217   144    70   235    67   77.79 
        6    81    32   164    13   121   145  5276     7    68    11   89.15 
        7    30    41   119    54   105    73     4  5612    37   190   89.58 
        8    60    89   134   294    86   249    86    72  4642   139   79.34 
        9    57    30    55   126   393    73    18   280   131  4786   80.45 
Precision 91.15 94.85 83.82 81.52 83.36 78.08 88.23 87.93 80.73 83.45 

Validation Accuracy: 0.857600

              0     1     2     3     4     5     6     7     8     9  Recall
        0   913     0     5    11     0    31    11     3     6     0   93.16 
        1     0  1104     3     6     2     3     5     3     9     0   97.27 
        2    23     4   841    26    28     6    31    23    48     2   81.49 
        3    12     2    38   828     3    61     2    23    31    10   81.98 
        4     4     4    10     1   852     8    14     9     7    73   86.76 
        5    23     5     9    65    14   686    23    13    42    12   76.91 
        6    23     6    16     0    31    28   839     1    10     4   87.58 
        7     3    10    36     9    13     7     0   908     7    35   88.33 
        8    14     9    18    37    18    45    18    11   777    27   79.77 
        9    11     3    11    14    72    18     0    36    16   828   82.06 
Precision 88.99 96.25 85.21 83.05 82.48 76.82 88.97 88.16 81.53 83.55 
```