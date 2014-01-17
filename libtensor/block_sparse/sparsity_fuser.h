#ifndef SPARSITY_FUSER_H
#define SPARSITY_FUSER_H

#include "block_loop.h"

namespace libtensor {

class sparsity_fuser 
{
private:
    std::vector<sparse_bispace_any_order> m_bispaces;
    std::vector<block_loop> m_loops;
    std::vector< sparse_block_tree_any_order > m_trees;
    std::vector<idx_list> m_loops_for_trees;
    std::vector<idx_list> m_trees_for_loops;
    std::vector<idx_pair_list> m_bispaces_and_index_groups_for_trees;
    std::vector<std::vector<idx_list> > m_sub_key_offsets_for_trees;
public:
    sparsity_fuser(std::vector< block_loop >& loops,
                   std::vector< sparse_bispace_any_order >& bispaces);

    std::vector<block_loop> get_loops() const { return m_loops; }
    std::vector<sparse_bispace_any_order> get_bispaces() const { return m_bispaces; }
    std::vector<sparse_block_tree_any_order> get_trees() const { return m_trees; } 

    idx_list  get_loops_for_tree(size_t tree_idx) const;
    idx_list get_trees_for_loop(size_t loop_idx) const;
    idx_pair_list get_bispaces_and_index_groups_for_tree(size_t tree_idx) const;
    std::vector<idx_list> get_sub_key_offsets_for_tree(size_t tree_idx) const;

    std::vector<off_dim_pair_list> get_offsets_and_sizes(size_t tree_idx) const;
    void fuse(size_t lhs_tree_idx,size_t rhs_tree_idx,const idx_list& loop_indices); 
};

} // namespace libtensor


#endif /* SPARSITY_FUSER_H */
