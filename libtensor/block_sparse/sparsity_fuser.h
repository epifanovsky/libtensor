#ifndef SPARSITY_FUSER_H
#define SPARSITY_FUSER_H

#include "block_loop.h"

namespace libtensor {

class sparsity_fuser 
{
private:
    std::vector< sparse_block_tree_any_order > m_trees;
    std::vector<idx_list> m_loops_for_trees;
    std::vector<idx_list> m_trees_for_loops;
    std::vector<idx_list> m_subspaces_for_loops;
public:
    sparsity_fuser(std::vector< block_loop >& loops,
                   std::vector< sparse_bispace_any_order >& bispaces);

    idx_list  get_loops_for_tree(size_t tree_idx) const;
    idx_list get_trees_for_loop(size_t loop_idx) const;
    void fuse(size_t lhs_tree_idx,size_t rhs_tree_idx,const idx_list& loop_indices); 
};

} // namespace libtensor


#endif /* SPARSITY_FUSER_H */
