#ifndef SPARSITY_FUSER_H
#define SPARSITY_FUSER_H

#include "block_loop.h"

namespace libtensor {

class sparsity_fuser 
{
private:
    std::vector< impl::sparse_block_tree_any_order > m_trees;
    std::vector< idx_list > m_trees_to_loops;
public:
    sparsity_fuser(std::vector< block_loop >& loops,
                   std::vector< sparse_bispace_any_order >& bispaces);

    idx_list get_loops_accessing_tree(size_t tree_idx) const;
};

} // namespace libtensor


#endif /* SPARSITY_FUSER_H */
