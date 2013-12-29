#include "sparsity_fuser.h"

using namespace std;

namespace libtensor {

sparsity_fuser::sparsity_fuser(vector< block_loop >& loops,
                               vector< sparse_bispace_any_order >& bispaces) : m_trees_for_loops(loops.size()),
                                                                               m_subspaces_for_loops(loops.size())
{
    //Extract all of the trees, tracking which loops access each one 
    //Also track the inverse - which trees a given loop accesses
    for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
    {
        const sparse_bispace_any_order& bispace = bispaces[bispace_idx];
        for(size_t tree_idx = 0; tree_idx < bispace.get_n_sparse_groups(); ++tree_idx)
        {
            m_loops_for_trees.push_back(idx_list());
            const sparse_block_tree_any_order& tree = bispace.get_sparse_group_tree(tree_idx);
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
                        /*m_trees_for_loops[loop_idx].push_back(all_trees_idx,);*/
                        /*size_t tree_subspace_looped = subspace_looped - min;*/
                        m_loops_for_trees.back().push_back(loop_idx);
                        m_subspaces_for_loops[loop_idx].push_back(subspace_looped - min);
                        m_trees_for_loops[loop_idx].push_back(m_trees.size() - 1);
                    }
                }
            }
        }
    }

    //Sort the trees to that the first one is accessed by the earliest loop that accesses any tree
}

idx_list sparsity_fuser::get_loops_for_tree(size_t tree_idx) const
{
    return m_loops_for_trees[tree_idx];
}

idx_list sparsity_fuser::get_trees_for_loop(size_t loop_idx) const
{
    return m_trees_for_loops[loop_idx];
}

void sparsity_fuser::fuse(size_t lhs_tree_idx,size_t rhs_tree_idx,const idx_list& loop_indices)
{
    //Find the subspaces lhs and rhs trees  for each loop
    /*idx_list lhs_subspaces;*/
    /*idx_list rhs_subspaces;*/
    for(size_t i = 0; i < loop_indices.size(); ++i)
    {
        size_t loop_idx = loop_indices[i];
        /*const idx_list& subspaces = m_subspaces_for_loop[loop_idx];*/
        /*idx_list::iterator lhs_subspace_pos = lower_bound(subspaces.begin(),subspaces.end(),lhs_tree_idx);*/
        /*idx_list::iterator rhs_subspace_pos = lower_bound(subspaces.begin(),subspaces.end(),rhs_tree_idx);*/

        /*lhs_subspaces.push_back(m_subspaces_for_loops*/

        //Merge the loops records corresponding to the lhs and rhs trees 
        /*idx_list& lhs_loops = m_loops_for_trees[lhs_tree_idx];*/
        /*const idx_list& rhs_loops = m_loops_for_trees[rhs_tree_idx];*/
        /*idx_list merged_loops(lhs_loops.size() + rhs_loops.size());*/
        /*idx_list::iterator it = set_union(lhs_loops.begin(),lhs_loops.end(),rhs_loops.begin(),rhs_loops.end(),merged_loops.begin());*/
        /*merged_loops.resize(it - merged_loops.begin());*/
        /*lhs_loops = merged_loops;*/
    }

    //Reassign all loops pointing to the rhs_tree to point to the lhs_tree and the appropriate subspace
    for(size_t loop_idx = 0; loop_idx < m_trees_for_loops.size(); ++loop_idx)
    {
        idx_list& trees = m_trees_for_loops[loop_idx];
        //Does this loop point to the rhs tree?
        idx_list::const_iterator rhs_tree_idx_it = find(trees.begin(),trees.end(),rhs_tree_idx);
        if(rhs_tree_idx_it != trees.end())
        {
            //If so, delete the reference and add one for the lhs tree in the appropriate position 
            trees.erase(rhs_tree_idx_it);
            idx_list::const_iterator lhs_tree_idx_it = lower_bound(trees.begin(),trees.end(),lhs_tree_idx);
            if((lhs_tree_idx_it == trees.end()) || (*lhs_tree_idx_it++ != lhs_tree_idx))
            {
                trees.insert(lhs_tree_idx_it,lhs_tree_idx);
            }
        }
    }
}

} // namespace libtensor
