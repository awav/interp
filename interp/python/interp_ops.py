import tensorflow as tf
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

__all__ = ["linear_interp", "cubic_interp_system", "regular_interp", "tri_diag_solve", "cubic_gather"]

ops = load_library.load_op_library(resource_loader.get_path_to_datafile('_interp_ops.so'))

linear_interp = ops.linear_interp
cubic_interp_system = ops.cubic_interp_system
regular_interp = ops.regular_interp
tri_diag_solve = ops.tri_diag_solve
cubic_gather = ops.cubic_gather
cubic_gather_rev = ops.cubic_gather_rev


@tf.RegisterGradient("TriDiagSolve")
def _tri_diag_solve_grad(op, *grads):
    diag, upper, lower, y = op.inputs
    x = op.outputs[0]
    bx = grads[0]
    by = tri_diag_solve(diag, lower, upper, bx)
    axes = tf.range(tf.rank(diag), tf.rank(y))
    bdiag = -tf.reduce_sum(x * by, axis=axes)

    n_inner = tf.shape(diag)[-1]
    axis = tf.rank(diag) - 1
    bupper = -tf.reduce_sum(
        tf.gather(x, tf.range(1, n_inner), axis=axis) *
        tf.gather(by, tf.range(n_inner-1), axis=axis),
        axis=axes)
    blower = -tf.reduce_sum(
        tf.gather(x, tf.range(n_inner-1), axis=axis) *
        tf.gather(by, tf.range(1, n_inner), axis=axis),
        axis=axes)

    return [bdiag, bupper, blower, by]


@tf.RegisterGradient("LinearInterp")
def _linear_interp_rev(op, *grads):
    t, x, y = op.inputs
    v, inds = op.outputs
    bv = grads[0]
    bt, by = linear_interp(t, x, y, inds, bv)
    return [bt, None, by]


@tf.RegisterGradient("CubicGather")
def _cubic_gather_rev(op, *grads):
    x = op.inputs[1]
    inds = op.outputs[-1]
    args = [x, inds] + list(grads)
    results = cubic_gather_rev(*args)
    return [tf.zeros_like(op.inputs[0])] + list(results)


@tf.RegisterGradient("RegularInterp")
def _regular_interp_rev(op, *grads):
    xi = op.inputs[-1]
    Z = op.outputs[0]
    dZ = op.outputs[1]
    bZ = grads[0]
    nx = tf.size(tf.shape(xi))
    axes = tf.range(nx, tf.size(tf.shape(Z)) + 1)
    bxi = tf.reduce_sum(dZ * tf.expand_dims(bZ, nx - 1), axis=axes)
    return tuple([None for i in range(len(op.inputs) - 1)] + [bxi])
