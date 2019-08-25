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