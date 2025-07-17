# CuTe Swizzle Preview

## Introduction

This application saves a LaTeX file with the shared memory bank ids for a given shared memory layout and swizzle configuration.

## Usages

```bash
$ ./build/examples/cute_swizzle_preview/cute_swizzle_preview --help
Print the shared memory bank ids for a given shared memory layout and swizzle configuration.

Options:

  --help                            If specified, displays this usage statement.

  --m=<int>                         Matrix on shared memory M dimension
  --n=<int>                         Matrix on shared memory N dimension
  --stride_m=<int>                  Matrix on shared memory M stride
  --stride_n=<int>                  Matrix on shared memory N stride
  --element_size=<int>              Element size in bytes
  --swizzle_num_mask_bits=<int>     Number of swizzle mask bits
  --swizzle_num_base=<int>          Number of swizzle base bits
  --swizzle_num_shift=<int>         Number of swizzle shift bits
  --latex_file_path=<string>        LaTeX file path

Examples:

$  --m=32 --n=64 --stride_m=64 --stride_n=1 --element_size=4 --swizzle_num_mask_bits=5 --swizzle_num_base=0 --swizzle_num_shift=6 --latex_file_path=shared_memory_bank_ids.tex
```
