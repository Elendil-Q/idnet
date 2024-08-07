# import os.path as osp
# from setuptools import setup, find_packages
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension
#
# ROOT = osp.dirname(osp.abspath(__file__))
#
#
# class get_pybind_include(object):
#     """Helper class to determine the pybind11 include path
#
#     The purpose of this class is to postpone importing pybind11
#     until it is actually installed, so that the ``get_include()``
#     method can be invoked. """
#
#     def __init__(self, user=False):
#         self.user = user
#
#     def __str__(self):
#         import pybind11
#         return pybind11.get_include(self.user)
#
#
# setup(
#     name="cuda_motion_compensation",
#     version="0.1",
#     packages=find_packages(),
#     ext_modules=[
#         CUDAExtension(
#             "warp_event",
#             sources=[
#                 "cuda_utils.cu",
#                 "warp_event.cpp",
#             ],
#             include_dirs=[
#                 "/usr/local/include/pybind11",
#                 "/usr/local/include",  # OpenCV的头文件路径
#                 "/usr/local/lib/",
#                 "/usr/local/cuda/include",
#                 "/usr/include",
#             ],
#             library_dirs=[
#                 r'/usr/include/',
#                 "/usr/lib/x86_64-linux-gnu/"
#             ],
#             libraries=[r'opencv_core', r'opencv_imgproc'],
#             extra_compile_args={
#                 "cxx": ["-O3","-std=c++17"],
#                 "nvcc": ["-O3"],
#             },
#         ),
#     ],
#     cmdclass={"build_ext": BuildExtension},
# )
