#ifndef SPARSITY_FUSER_H
#define SPARSITY_FUSER_H

#include "block_loop.h"

namespace libtensor {

class sparsity_fuser 
{
private:
    std::vector<sparse_bispace_any_order> m_bispaces;
    std::vector<block_loop> m_loops;
    std::vector< sparsity_data > m_trees;
    std::vector<idx_list> m_loops_for_trees;
    std::vector<idx_list> m_trees_for_loops;
    std::vector<idx_pair_list> m_bispaces_and_index_groups_for_trees;
    std::vector<std::vector<idx_list> > m_sub_key_offsets_for_trees;

    const idx_list m_direct_tensors;
    const std::map<size_t,idx_pair> m_batches;
public:
    sparsity_fuser(const std::vector< block_loop >& loops,
                   const std::vector< sparse_bispace_any_order >& bispaces,
                   const idx_list& direct_tensors = idx_list(),
                   const std::map<size_t,idx_pair>& batches = (std::map<size_t,idx_pair>()));

    std::vector<block_loop> get_loops() const { return m_loops; }
    std::vector<sparse_bispace_any_order> get_bispaces() const { return m_bispaces; }
    std::vector<sparsity_data> get_trees() const { return m_trees; } 

    idx_list get_direct_tensors() const { return m_direct_tensors; }
    std::map<size_t,idx_pair> get_batches() const { return m_batches; }

    const idx_list&  get_loops_for_tree(size_t tree_idx) const;
    const idx_list& get_trees_for_loop(size_t loop_idx) const;
    const idx_pair_list& get_bispaces_and_index_groups_for_tree(size_t tree_idx) const;
    const std::vector<idx_list>& get_sub_key_offsets_for_tree(size_t tree_idx) const;

    void fuse(size_t lhs_tree_idx,size_t rhs_tree_idx,const idx_list& loop_indices); 
};

} // namespace libtensor


#endif /* SPARSITY_FUSER_H */
