#ifndef GET_BATCHES_H
#define GET_BATCHES_H

#include "sparse_bispace.h"

namespace libtensor {

std::vector<idx_pair> get_batches(const std::vector<sparse_bispace_any_order>& bispaces,
                                  const std::vector<idx_pair>& batched_subspaces,
                                  size_t max_n_elem);

} // namespace libtensor

#endif /* GET_BATCHES_H */
