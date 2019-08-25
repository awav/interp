licenses(["notice"])  # Apache 2.0

package(default_visibility = ["//visibility:public"])

load("@local_config_cuda//cuda:build_defs.bzl", "if_cuda", "if_cuda_is_configured")

cc_library(
    name = "cuda",
    data = [
        "@local_config_cuda//cuda:cudart",
    ],
    linkopts = select({
        "@local_config_cuda//cuda:darwin": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib",
        ],
        "//conditions:default": [
            "-Wl,-rpath,../local_config_cuda/cuda/lib64",
            "-Wl,-rpath,../local_config_cuda/cuda/extras/CUPTI/lib64",
        ],
    }),
    deps = [
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_library(
    name = "interp_ops_gpu",
    srcs = [
        "cc/linear/cuda_utils.h",
        "cc/linear/linear_op.cc.cu",
        "cc/linear/linear_op.h",
    ],
    copts = if_cuda_is_configured([
        "-DGOOGLE_CUDA=1",
        "-x cuda",
        "-nvcc_options=relaxed-constexpr",
        "-nvcc_options=ftz=true",
    ]),
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([
        ":cuda",
        "@local_config_cuda//cuda:cuda_headers",
    ]),
    alwayslink = 1,
)

cc_binary(
    name = "python/_interp_ops.so",
    srcs = [
        "cc/cubic/cubic_gather.cc",
        "cc/cubic/cubic_gather_rev.cc",
        "cc/cubic/get_system.cc",
        "cc/cubic/search_sorted_op.cc",
        "cc/cubic/tri_diag_solve_op.cc",
        "cc/linear/cuda_utils.h",
        "cc/linear/interp.h",
        "cc/linear/linear_op.cc",
        "cc/linear/linear_op.h",
        "cc/linear/linear_rev_op.cc",
        "cc/regular/regular_interp.cc",
    ],
    copts = [
        "-pthread",
        "-std=c++11",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ] + if_cuda(["-DGOOGLE_CUDA=1"]),
    linkshared = 1,
    deps = [
        "@local_config_tf//:libtensorflow_framework",
        "@local_config_tf//:tf_header_lib",
    ] + if_cuda_is_configured([":interp_ops_gpu"]),
)

py_library(
    name = "interp_ops_py",
    srcs = ([
        "python/interp_ops.py",
    ]),
    data = [
        ":python/_interp_ops.so",
    ],
    srcs_version = "PY2AND3",
)

py_library(
    name = "interp_py",
    srcs = ([
        "__init__.py",
        "python/__init__.py",
        "python/interp.py",
    ]),
    srcs_version = "PY2AND3",
    deps = [
        ":interp_ops_py",
    ],
)

# py_test(
#     name = "zero_out_ops_py_test",
#     srcs = [
#         "python/ops/zero_out_ops_test.py"
#     ],
#     main = "python/ops/zero_out_ops_test.py",
#     deps = [
#         ":zero_out_ops_py",
#     ],
#     srcs_version = "PY2AND3",
# )
