Multivariate Linear Interpolation in TensorFlow
===


<!-- This module implements a custom TensorFlow operation that replicates the
[RegularGridInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html)
from SciPy. -->

To install, run:

```bash
bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip install artifacts/interp-*-py3-none-any.whl
```

```python
import numpy as np
import tensorflow as tf
import interp

X = np.asarray(sorted(np.random.rand(10)))
Y = np.random.rand(10)
xstar = np.linspace(0, 1, 20)

li = interp.LinearInterpolator(X, Y)
ystar_tensor = li.evaluate(xstar)
session = tf.Session()
session.run(ystar_tensor)
```