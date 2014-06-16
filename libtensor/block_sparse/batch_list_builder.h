#ifndef BATCH_LIST_BUILDER_H
#define BATCH_LIST_BUILDER_H

#include "labeled_bispace.h"
#include "subspace_iterator.h"

namespace libtensor {

class batch_list_builder
{
private:
    std::vector< std::vector<subspace_iterator> > m_iter_groups;
    size_t m_end_idx;
public:
    batch_list_builder(const std::vector< std::vector<labeled_bispace> >& labeled_bispace_groups,
                       const letter& batched_index);

    idx_pair_list get_batch_list(size_t max_n_elem);
};

} // namespace libtensor

#endif /* BATCH_LIST_BUILDER_H */
