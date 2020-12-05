#pragma once

namespace libtensor {

struct BlasSequential {
  BlasSequential();
  ~BlasSequential();
  int blas_num_threads;
};

}  // namespace libtensor

