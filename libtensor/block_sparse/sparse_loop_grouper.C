#include "sparse_loop_grouper.h"

using namespace std;

namespace libtensor {

const char* sparse_loop_grouper::k_clazz = "sparse_loop_grouper";

sparse_loop_grouper::sparse_loop_grouper(const sparsity_fuser& sf) 
{

    //Check that all loops are fused appropriately for grouping
    vector<block_loop> loops = sf.get_loops();
    for(size_t loop_idx = 0; loop_idx < loops.size(); ++loop_idx)
    {
        if(sf.get_trees_for_loop(loop_idx).size() > 1)
        {
            throw bad_parameter(g_ns,k_clazz,"sparse_loop_grouper(...)",__FILE__,__LINE__,
                    "loops not fully fused");
        }
    } 

    vector<sparse_bispace_any_order> bispaces = sf.get_bispaces();
    vector<sparse_block_tree_any_order> trees = sf.get_trees(); 

    //Group the loops 
    size_t cur_loop_idx = 0;
    while(cur_loop_idx < loops.size())
    {
        idx_list trees_for_loop = sf.get_trees_for_loop(cur_loop_idx);
        idx_pair_list baig;
        idx_list loops_in_grp;
        if(trees_for_loop.size() > 0)
        {
            //Sparse loop group


            //Process the offsets for this loop group
            const sparse_block_tree_any_order& cur_tree = trees[m_offsets_and_sizes.size()];
            vector<off_dim_pair_list> cur_offsets_and_sizes;
            for(sparse_block_tree_any_order::const_iterator it = cur_tree.begin(); it != cur_tree.end(); ++it)
            {
                cur_offsets_and_sizes.push_back(*it); 
            }

            baig = sf.get_bispaces_and_index_groups_for_tree(trees_for_loop[0]);

            for(size_t loop_rel_idx = 0; loop_rel_idx < cur_tree.get_order(); ++loop_rel_idx)
            {
                loops_in_grp.push_back(cur_loop_idx + loop_rel_idx);
            }
            m_offsets_and_sizes.push_back(cur_offsets_and_sizes);
            cur_loop_idx += cur_tree.get_order();
        }

        //Fill in the bispace/index group pairs and sizes and offsets that are touched in DENSE TENSORS
        //These are not accounted for in the bispaces and index groups associated with a given tree
        for(size_t loop_rel_idx = 0; loop_rel_idx < loops_in_grp.size(); ++loop_rel_idx)
        {
            const block_loop& loop = loops[loops_in_grp[loop_rel_idx]];
            for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
            {
                const sparse_bispace_any_order& bispace = bispaces[bispace_idx]; 
                if(!loop.is_bispace_ignored(bispace_idx))
                {
                    //Append bispaces/index groups information
                    size_t subspace_idx = loop.get_subspace_looped(bispace_idx);
                    idx_pair cur_pair(bispace_idx,bispace.get_index_group_containing_subspace(subspace_idx));
                    if(find(baig.begin(),baig.end(),cur_pair) == baig.end())
                    {
                        baig.push_back(cur_pair);
                    }

                    //Append offsets and sizes information
                }
            }
        }
        m_bispaces_and_index_groups.push_back(baig);
    }
}

vector<idx_pair_list> sparse_loop_grouper::get_bispaces_and_index_groups() const
{
    return m_bispaces_and_index_groups;
}

vector< vector<off_dim_pair_list> > sparse_loop_grouper::get_offsets_and_sizes() const
{
    return m_offsets_and_sizes;
}

} // namespace libtensor
