#include "sparsity_fuser.h"

using namespace std;

namespace libtensor {

sparsity_fuser::sparsity_fuser(vector< block_loop >& loops,
                               vector< sparse_bispace_any_order >& bispaces) : m_loops(loops), 
                                                                               m_bispaces(bispaces),
                                                                               m_trees_for_loops(loops.size())
{
    //Extract all of the trees, tracking which loops access each one 
    //Also track the inverse - which trees a given loop accesses
    for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
    {
        const sparse_bispace_any_order& bispace = bispaces[bispace_idx];
        for(size_t tree_idx = 0; tree_idx < bispace.get_n_sparse_groups(); ++tree_idx)
        {
            sparse_block_tree_any_order tree = bispace.get_sparse_group_tree(tree_idx);
            m_sub_key_offsets_for_trees.push_back(vector<idx_list>(1,idx_list()));
            for(size_t tree_sub_idx = 0; tree_sub_idx < tree.get_order(); ++tree_sub_idx)
            {
                m_sub_key_offsets_for_trees.back().back().push_back(tree_sub_idx);
            }

            m_loops_for_trees.push_back(idx_list());
            size_t min = bispace.get_sparse_group_offset(tree_idx);
            size_t max = min + tree.get_order();

            idx_list perm_entries(tree.get_order());
            size_t loops_accessing_tree_idx = 0;
            for(size_t loop_idx = 0; loop_idx < loops.size(); ++loop_idx)
            {
                const block_loop& loop = loops[loop_idx];
                if(!loop.is_bispace_ignored(bispace_idx))
                {
                    //Does this loop touch this tree?
                    size_t subspace_looped = loop.get_subspace_looped(bispace_idx);
                    if((min <= subspace_looped) && (subspace_looped < max))
                    {
                        m_loops_for_trees.back().push_back(loop_idx);
                        m_trees_for_loops[loop_idx].push_back(m_trees.size());
                        perm_entries[loops_accessing_tree_idx] = subspace_looped - min;
                        ++loops_accessing_tree_idx;
                    }
                }
            }

            //Save the tree permuted into loop order
            m_trees.push_back(tree.permute(runtime_permutation(perm_entries)));

            //Save the index group/bispace mapping information for the tree
            size_t index_group = bispace.get_index_group_containing_subspace(min);
            m_bispaces_and_index_groups_for_trees.push_back(idx_pair_list(1,idx_pair(bispace_idx,index_group)));
        }
    }
}

idx_list sparsity_fuser::get_loops_for_tree(size_t tree_idx) const
{
    return m_loops_for_trees[tree_idx];
}

idx_list sparsity_fuser::get_trees_for_loop(size_t loop_idx) const
{
    return m_trees_for_loops[loop_idx];
}

idx_pair_list sparsity_fuser::get_bispaces_and_index_groups_for_tree(size_t tree_idx) const
{
    return m_bispaces_and_index_groups_for_trees[tree_idx];
}

vector<off_dim_pair_list> sparsity_fuser::get_offsets_and_sizes(size_t tree_idx) const
{
    vector<off_dim_pair_list> offsets_and_sizes;
    const sparse_block_tree_any_order& tree = m_trees[tree_idx];
    for(sparse_block_tree_any_order::const_iterator it = tree.begin(); it != tree.end(); ++it)
    {
        offsets_and_sizes.push_back(*it);
    }
    return offsets_and_sizes;
}

vector<idx_list> sparsity_fuser::get_sub_key_offsets_for_tree(size_t tree_idx) const
{
    return m_sub_key_offsets_for_trees[tree_idx];
}

void sparsity_fuser::fuse(size_t lhs_tree_idx,size_t rhs_tree_idx,const idx_list& loop_indices)
{
    //Find the subspaces lhs and rhs trees  for each loop
    idx_list lhs_subspaces;
    idx_list rhs_subspaces;
    idx_list& lhs_tree_loops = m_loops_for_trees[lhs_tree_idx];
    idx_list& rhs_tree_loops = m_loops_for_trees[rhs_tree_idx];
    for(size_t i = 0; i < loop_indices.size(); ++i)
    {
        size_t loop_idx = loop_indices[i];
        //The tree subspace corresponding to each loop corresponds to its position in the m_loops_for_trees entry
        idx_list::const_iterator lhs_it = find(lhs_tree_loops.begin(),lhs_tree_loops.end(),loop_idx);
        idx_list::const_iterator rhs_it = find(rhs_tree_loops.begin(),rhs_tree_loops.end(),loop_idx);

        lhs_subspaces.push_back(lhs_it - lhs_tree_loops.begin());
        rhs_subspaces.push_back(rhs_it - rhs_tree_loops.begin());
    }

    //Reassign all loops pointing to the rhs_tree to point to the lhs_tree and the appropriate subspace
    //Also, now that we have one less tree, decrement all tree indices greater than the rhs_tree_idx
    for(size_t loop_idx = 0; loop_idx < m_trees_for_loops.size(); ++loop_idx)
    {
        idx_list& trees = m_trees_for_loops[loop_idx];
        //Does this loop point to the rhs tree?
        idx_list::iterator rhs_tree_idx_it = find(trees.begin(),trees.end(),rhs_tree_idx);
        if(rhs_tree_idx_it != trees.end())
        {
            //If so, delete the reference and add one for the lhs tree in the appropriate position 
            trees.erase(rhs_tree_idx_it);
            idx_list::iterator lhs_tree_idx_it = lower_bound(trees.begin(),trees.end(),lhs_tree_idx);
            if((lhs_tree_idx_it == trees.end()) || (*lhs_tree_idx_it != lhs_tree_idx))
            {
                trees.insert(lhs_tree_idx_it,lhs_tree_idx);
            }
        }

        //Decrement indices larger than rhs_tree_idx
        for(size_t rel_tree_idx = 0; rel_tree_idx < trees.size(); ++rel_tree_idx)
        {
            if(trees[rel_tree_idx] > rhs_tree_idx)
            {
                --trees[rel_tree_idx];
            }
        }
    }
    
    //The lhs tree now points to all the bispaces and index groups previously associated with the rhs tree 
    const idx_pair_list& rhs_baig = m_bispaces_and_index_groups_for_trees[rhs_tree_idx];  
    idx_pair_list& lhs_baig = m_bispaces_and_index_groups_for_trees[lhs_tree_idx];  
    for(size_t baig_idx = 0; baig_idx < rhs_baig.size(); ++baig_idx)
    {
        lhs_baig.push_back(rhs_baig[baig_idx]);
    }

    //The lhs tree is now associated with all loops that pointed to the RHS tree
    for(size_t loop_rel_idx = 0; loop_rel_idx < rhs_tree_loops.size(); ++loop_rel_idx)
    {
        size_t rhs_loop = rhs_tree_loops[loop_rel_idx];
        if(find(loop_indices.begin(),loop_indices.end(),rhs_loop) == loop_indices.end())
        {
            lhs_tree_loops.push_back(rhs_loop);
        }
    }

    //The portions of the key associated with the rhs tree are now the corresponding lhs subspaces  
    const vector<idx_list>& rhs_sub_key_offsets = m_sub_key_offsets_for_trees[rhs_tree_idx];
    idx_list rhs_unfused_inds;
    for(size_t rhs_subspace_idx = 0; rhs_subspace_idx < m_trees[rhs_tree_idx].get_order(); ++rhs_subspace_idx)
    {
        if(find(rhs_subspaces.begin(),rhs_subspaces.end(),rhs_subspace_idx) == rhs_subspaces.end())
        {
            rhs_unfused_inds.push_back(rhs_subspace_idx);
        }
    }
    for(size_t idx_grp_idx = 0; idx_grp_idx < rhs_sub_key_offsets.size(); ++idx_grp_idx)
    {
        const idx_list& rhs_grp_sub_key_offsets = rhs_sub_key_offsets[idx_grp_idx];
        idx_list lhs_grp_sub_key_offsets;
        for(size_t rhs_subspace_rel_idx = 0; rhs_subspace_rel_idx < rhs_grp_sub_key_offsets.size(); ++rhs_subspace_rel_idx)
        {
            //If the index was fused, it belongs to the lhs subspace with which it was fused  
            size_t rhs_subspace_idx = rhs_grp_sub_key_offsets[rhs_subspace_rel_idx];
            idx_list::iterator rhs_fused_pos = find(rhs_subspaces.begin(),rhs_subspaces.end(),rhs_subspace_idx);
            if(rhs_fused_pos != rhs_subspaces.end()) 
            {
                lhs_grp_sub_key_offsets.push_back(lhs_subspaces[distance(rhs_subspaces.begin(),rhs_fused_pos)]);
            }
            else
            {
                //index was not fused - it is now at position [lhs tree size] + [unfused position]
                idx_list::iterator rhs_unfused_pos = find(rhs_unfused_inds.begin(),rhs_unfused_inds.end(),rhs_subspace_idx);
                lhs_grp_sub_key_offsets.push_back(m_trees[lhs_tree_idx].get_order() + distance(rhs_unfused_inds.begin(),rhs_unfused_pos));
            }
        }
        m_sub_key_offsets_for_trees[lhs_tree_idx].push_back(lhs_grp_sub_key_offsets);
    }

    //Fuse the trees, delete the rhs tree 
    m_trees[lhs_tree_idx] = m_trees[lhs_tree_idx].fuse(m_trees[rhs_tree_idx],lhs_subspaces,rhs_subspaces);
    m_trees.erase(m_trees.begin()+rhs_tree_idx);

    //Remove metadata associated with the rhs tree
    m_sub_key_offsets_for_trees.erase(m_sub_key_offsets_for_trees.begin()+rhs_tree_idx);
    m_loops_for_trees.erase(m_loops_for_trees.begin()+rhs_tree_idx);
    m_bispaces_and_index_groups_for_trees.erase(m_bispaces_and_index_groups_for_trees.begin()+rhs_tree_idx);
}

} // namespace libtensor
