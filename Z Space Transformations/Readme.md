### Finish the `z_transform(X, degree = 2)` function in the enclose `utils.py` file. You only need to submit the `utils.py` file. We (the TA and myself) will use our own main module to test your code. 

This `z_transform(X, degree = 2)` function takes and transforms an $n \times d$ matrix, which represents the $n$ samples from the $X$ space where each sample has $d$ features, into another matrix $Z$ of dimension size $n \times d'$ that represents the transformed samples in the $Z$ space, where each sample had $d'$ features.

If your code is correct, it will produce a new matrix such that the following is always held between the values of $d$ and $d'$:

$$
d' = \sum_{i=1}^{degree}\binom{i+d-1}{d-1}
$$

Given the matrix $X$ and the degree of the $Z$ space that you want to go to, the matrix $Z$ is determined. Your code inside the body of `z_transform(X, degree = 2)` is to compute $Z$ and return it. 

For example: if 

```
    X =
     [[1 2]
     [2 3]
     [3 4]]
```

-  `z_transform(X, degree = 1)` should produce and return 

```
    [[1 2]
     [2 3]
     [3 4]]
```

- `z_transform(X, degree = 2)` should produce and return

```
    [[ 1  2  1  2  4]
     [ 2  3  4  6  9]
     [ 3  4  9 12 16]]
```

- `z_transform(X, degree = 3)` should produce and return 

```
    [[ 1  2  1  2  4  1  2  4  8]
     [ 2  3  4  6  9  8 12 18 27]
     [ 3  4  9 12 16 27 36 48 64]]
```
 
- `z_transform(X, degree = 4)` should produce and return 

```
    [[  1   2   1   2   4   1   2   4   8   1   2   4   8  16]
     [  2   3   4   6   9   8  12  18  27  16  24  36  54  81]
     [  3   4   9  12  16  27  36  48  64  81 108 144 192 256]]
```

- `z_transform(X, degree = 5)` should produce and return

```
    [[   1    2    1    2    4    1    2    4    8    1    2    4    8   16   1    2    4    8   16   32]
     [   2    3    4    6    9    8   12   18   27   16   24   36   54   81  32   48   72  108  162  243]
     [   3    4    9   12   16   27   36   48   64   81  108  144  192  256 243  324  432  576  768 1024]]
```

Make sure your code works for any $n\geq 1$, any $d\geq 1$, and any $degree \geq 1$. 



After you successfully complete the `z_transform` function, you may use it to replace the empty `Z-transform` function that I shared for prog2 and play with it for classification for data that are (nearly) linearly separable in various Z spaces. 

I also enclosed the `test_pla-v2.ipynb` file that you can use (but don't have to if you want to have your own) for quick testing/playing with the code I shared from prog2. This notebook for testing/playing only supports degree up to 4 (due to my not having time to code out the generic form for the plotting part; you feel free to change it), but the `z_transform` function itself supports any `degree`. 



