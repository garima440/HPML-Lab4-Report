ECE-GY 9143 - High Performance Machine Learning

Homework Assignment 4

Parijat Dube and Zehra Sura

###Part-A: CUDA Matrix Operations 

The purpose of this exercise is for you to learn how to write programs using the CUDA
programming interface, how to run such programs using an NVIDIA graphics processor,
and how to think about the factors that govern the performance of programs running
within the CUDA environment. For this assignment, you will modify two provided CUDA
kernels. The first is part of a program that performs vector addition. The second is part
of a program to perform matrix multiplication. The provided vector addition program
does not coalesce memory accesses. You will modify it to coalesce memory access. You
will modify the matrix multiplication program to investigate its performance and how it
can be optimized by changing how the task is parallelized. You will create an improved
version of the matrix multiplication program and empirically test the time it takes to
run. You will analyze the results of your timing tests.

###Part-B: CUDA Unified Memory

In this problem we will compare vector operations executed on host vs on GPU to quantify
the speed-up.

###Part-C: Convolution in CUDA

We use three Approaches to implement convolution in CUDA.
