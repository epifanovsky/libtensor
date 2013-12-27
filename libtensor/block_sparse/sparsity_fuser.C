#include "sparsity_fuser.h"

namespace libtensor {

sparsity_fuser::sparsity_fuser(std::vector< block_loop >& loops,
                               std::vector< sparse_bispace_any_order >& bispaces)
{
    //Extract all of the trees, tracking which loops access each one 
    //Also track the inverse - which trees a given loop accesses
    for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
    {
        const sparse_bispace_any_order& bispace = bispaces[bispace_idx];
        for(size_t tree_idx = 0; tree_idx < bispace.get_n_sparse_groups(); ++tree_idx)
        {
            m_trees_to_loops.push_back(idx_list());
            const impl::sparse_block_tree_any_order& tree = bispace.get_sparse_group_tree(tree_idx);
            m_trees.push_back(tree);
            size_t min = bispace.get_sparse_group_offset(tree_idx);
            size_t max = min + tree.get_order();

            for(size_t loop_idx = 0; loop_idx < loops.size(); ++loop_idx)
            {
                const block_loop& loop = loops[loop_idx];
                if(!loop.is_bispace_ignored(bispace_idx))
                {
                    //Does this loop touch this tree?
                    size_t subspace_looped = loop.get_subspace_looped(bispace_idx);
                    if((min <= subspace_looped) && (subspace_looped < max))
                    {
                        /*m_loops_to_trees[loop_idx].push_back(all_trees_idx,);*/
                        /*size_t tree_subspace_looped = subspace_looped - min;*/
                        m_trees_to_loops.back().push_back(loop_idx);
                    }
                }
            }
        }
    }
}

idx_list sparsity_fuser::get_loops_accessing_tree(size_t tree_idx) const
{
    return m_trees_to_loops[tree_idx];
}

} // namespace libtensor
