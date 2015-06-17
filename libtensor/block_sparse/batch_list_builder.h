#ifndef BATCH_LIST_BUILDER_H
#define BATCH_LIST_BUILDER_H

#include "subspace_iterator.h"
#include "connectivity.h"

namespace libtensor {

class batch_list_builder
{
private:
    std::vector< std::vector<subspace_iterator> > m_iter_grps;
    size_t m_end_idx;
public:
    batch_list_builder(const std::vector< std::vector<sparse_bispace_any_order> >& bispace_grps,
                       const std::vector<idx_list>& batched_subspace_grps);

    idx_pair_list get_batch_list(size_t max_n_elem);
    std::vector< std::vector<size_t> > get_batch_array_size_grps(const idx_pair_list& batch_list);
};

} // namespace libtensor

#endif /* BATCH_LIST_BUILDER_H */
