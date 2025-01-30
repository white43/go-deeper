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

Epoch 1 (8.71 sec), val_acc: 0.4819, lr: 0.1000
Epoch 2 (8.44 sec), val_acc: 0.6509, lr: 0.1000
Epoch 3 (8.53 sec), val_acc: 0.7414, lr: 0.1000
Epoch 4 (8.43 sec), val_acc: 0.7790, lr: 0.1000
Epoch 5 (8.84 sec), val_acc: 0.8005, lr: 0.1000
Epoch 6 (8.95 sec), val_acc: 0.8186, lr: 0.1000
Epoch 7 (8.53 sec), val_acc: 0.8291, lr: 0.1000
Epoch 8 (8.49 sec), val_acc: 0.8373, lr: 0.1000
Epoch 9 (8.71 sec), val_acc: 0.8459, lr: 0.1000
Epoch 10 (8.47 sec), val_acc: 0.8531, lr: 0.1000

Training Accuracy: 0.852067

              0     1     2     3     4     5     6     7     8     9  Recall
        0  5442     0    71    43     9   174    45    31   101     7   91.88 
        1     0  6427    83    44    13    62    14    21    70     8   95.33 
        2    97    68  4919   149   109    51   168   126   226    45   82.56 
        3    30    24   247  5046    17   386    57    75   161    88   82.30 
        4    35    27    55    25  4964    48   122    35   101   430   84.97 
        5   134    48    88   307    70  4294   148    57   215    60   79.21 
        6   108    57   156    24   110   149  5264     1    48     1   88.95 
        7    48    68   127    54   118    28     7  5493    40   282   87.68 
        8    40   106   156   228    78   402    98    50  4565   128   78.02 
        9    46    32    49   114   416   120     9   349   104  4710   79.17 
Precision 91.00 93.73 82.66 83.63 84.08 75.15 88.74 88.06 81.07 81.79 

Validation Accuracy: 0.853100

              0     1     2     3     4     5     6     7     8     9  Recall
        0   913     0    10     6     3    25    10     5     7     1   93.16 
        1     0  1085    10     9     1     7     5     0    14     4   95.59 
        2    23     8   838    33    24     7    24    20    47     8   81.20 
        3     3     2    27   834     1    71     8    11    43    10   82.57 
        4     5     2     7     8   819     7    32     7    18    77   83.40 
        5    20     6     8    51    11   723    20    12    32     9   81.05 
        6    19     7    26     6    19    35   841     0     4     1   87.79 
        7     9    19    35    16    13     2     4   890     5    35   86.58 
        8     6    12    22    34    18    64    17    10   770    21   79.06 
        9    16     6     6    16    69    24     1    40    13   818   81.07 
Precision 90.04 94.59 84.73 82.33 83.74 74.92 87.42 89.45 80.80 83.13
```