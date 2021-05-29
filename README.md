# models

**2021 Spring**, **Capstone design course**, in Sejon Univ.

****

# run experiment

make experiment with three models below

1. faceEstimator
2. handEstimator
3. obejctDetector

when you want to use these models in other python file

first import models,

~~~python
from models import faceEstimator

from models import handEstimator

from models import objectDetector
~~~

and call models like,
~~~python
faceEstimator()

handEstimator()

objectDetector()
~~~

## Arguments

- `--model` : Select model (One of 'faceEstimator', 'handEstimator', 'objectDetector')
