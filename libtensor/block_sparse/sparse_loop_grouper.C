#include "sparse_loop_grouper.h"

//TODO REMOVE
#include <iostream>

//TODO: DEBUG REMOVE for read_timer
#include "sparse_loop_list.h"

using namespace std;

namespace libtensor {

const char* sparse_loop_grouper::k_clazz = "sparse_loop_grouper";



sparse_loop_grouper::sparse_loop_grouper(const sparsity_fuser& sf)
{
    /*double seconds = read_timer<double>();*/

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
    /*double bispace_tree_seconds = read_timer<double>();*/
    vector<sparse_bispace_any_order> bispaces = sf.get_bispaces();
    vector<sparse_block_tree_any_order> trees = sf.get_trees(); 
    /*std::cout << "tree bispace copy outside time: " << read_timer<double>() - bispace_tree_seconds << "\n";*/

    /*double get_batches_seconds = read_timer<double>();*/
    map<size_t,idx_pair> batches = sf.get_batches();
    /*std::cout << "get_batches outside time: " << read_timer<double>() - get_batches_seconds << "\n";*/

    //Group the loops 
    vector<idx_list> trees_for_groups;
    for(size_t loop_idx = 0; loop_idx < loops.size(); ++loop_idx)
    {
        const block_loop& loop = loops[loop_idx];
        size_t cur_grp_idx;
        idx_list trees_for_loop = sf.get_trees_for_loop(loop_idx);

        block_list loop_block_list;
        idx_pair_list tree_baig;
        vector<off_dim_pair_list> sparse_offsets_and_sizes;
        //Sparse loop?
        if(trees_for_loop.size() == 1)
        {
            //Is this loop part of a group that we have already started?
            size_t tree_idx = trees_for_loop[0];
            const sparse_block_tree_any_order& tree = trees[tree_idx];
            bool found = false;
            for(size_t grp_idx = 0; grp_idx < trees_for_groups.size(); ++grp_idx)
            {
                const idx_list& this_grp_trees = trees_for_groups[grp_idx];
                idx_list::const_iterator grp_pos = find(this_grp_trees.begin(),this_grp_trees.end(),tree_idx);
                if(grp_pos != this_grp_trees.end())
                {
                    //Yes, we are part of an existing group
                    cur_grp_idx = grp_idx;
                    m_loops_for_groups[cur_grp_idx].push_back(loop_idx);
                    found = true;
                    break;
                }
            }
            if(!found)
            {
                //No, make a new sparse group
                m_loops_for_groups.push_back(idx_list(1,loop_idx));
                trees_for_groups.push_back(idx_list(1,tree_idx));
                m_bispaces_and_subspaces.push_back(idx_pair_list());
                m_bispaces_and_index_groups.push_back(idx_pair_list());
                m_offsets_and_sizes.push_back(vector<off_dim_pair_list>(tree.get_n_entries()));
                m_block_dims.push_back(vector<dim_list>(tree.get_n_entries()));
                cur_grp_idx = trees_for_groups.size() - 1;
            }
            idx_list loops_for_tree = sf.get_loops_for_tree(tree_idx);
            size_t tree_subspace_idx = distance(loops_for_tree.begin(),find(loops_for_tree.begin(),loops_for_tree.end(),loop_idx));
            tree_baig = sf.get_bispaces_and_index_groups_for_tree(tree_idx);

            /*double iter_sec = read_timer<double>();*/
            for(sparse_block_tree_any_order::const_iterator it = tree.begin(); it != tree.end(); ++it)
            {
                loop_block_list.push_back(it.key()[tree_subspace_idx]);
                sparse_offsets_and_sizes.push_back(*it);
            }
            /*std::cout  << "Offset copy time: " << read_timer<double>() - iter_sec << "\n";*/
        }
        else
        {
            //Dense loop - automatically a new group unto itself
            trees_for_groups.push_back(idx_list());
            m_loops_for_groups.push_back(idx_list(1,loop_idx));
            m_bispaces_and_subspaces.push_back(idx_pair_list());
            m_bispaces_and_index_groups.push_back(idx_pair_list());
            for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
            {
                if(!loop.is_bispace_ignored(bispace_idx))
                {
                    size_t subspace_idx = loop.get_subspace_looped(bispace_idx);
                    const sparse_bispace<1>& subspace = bispaces[bispace_idx][subspace_idx];
                    m_block_dims.push_back(vector<dim_list>(subspace.get_n_blocks()));

                    //If the loop is batched, keep only blocks from the current batch
                    size_t min_block_idx,max_block_idx;
                    if(batches.find(loop_idx) != batches.end())
                    {
                        min_block_idx = batches[loop_idx].first;
                        max_block_idx = batches[loop_idx].second;
                    }
                    else
                    {
                        min_block_idx = 0;
                        max_block_idx = subspace.get_n_blocks();
                    }

                    for(size_t block_idx = min_block_idx; block_idx < max_block_idx; ++block_idx)
                        loop_block_list.push_back(block_idx);
                    break;
                }
            }
            m_offsets_and_sizes.push_back(vector<off_dim_pair_list>(loop_block_list.size()));
            cur_grp_idx = trees_for_groups.size() - 1;
        }

        //Append the bispaces/subspaces traversed by this loop to the appropriate group
        for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
        {
            const sparse_bispace_any_order& bispace = bispaces[bispace_idx];
            if(!loop.is_bispace_ignored(bispace_idx))
            {
                size_t subspace_idx = loop.get_subspace_looped(bispace_idx);
                m_bispaces_and_subspaces[cur_grp_idx].push_back(idx_pair(bispace_idx,subspace_idx));
                const sparse_bispace<1>& subspace = bispace[subspace_idx];
                for(size_t block_rel_idx = 0; block_rel_idx < loop_block_list.size(); ++block_rel_idx)
                {
                    size_t size = subspace.get_block_size(loop_block_list[block_rel_idx]);
                    m_block_dims[cur_grp_idx][block_rel_idx].push_back(size);
                }

                //Append the bispace/index group pair if it is unique
                //Also append offset and size information associated with the index group
                size_t idx_grp = bispace.get_index_group_containing_subspace(subspace_idx);
                idx_pair_list& this_grp_baig = m_bispaces_and_index_groups[cur_grp_idx];
                idx_pair baig_entry(bispace_idx,idx_grp);
                if(find(this_grp_baig.begin(),this_grp_baig.end(),baig_entry) == this_grp_baig.end()) 
                {
                    this_grp_baig.push_back(baig_entry);

                    //Handle sparse offsets
                    bool treated_sparse = false;
                    if(tree_baig.size() > 0)
                    {
                        idx_pair_list::iterator baig_pos = find(tree_baig.begin(),tree_baig.end(),baig_entry);
                        if(baig_pos != tree_baig.end())
                        {
                            size_t baig_idx = distance(tree_baig.begin(),baig_pos);
                            for(size_t off_sz_idx = 0; off_sz_idx < sparse_offsets_and_sizes.size(); ++off_sz_idx)
                            {
                                m_offsets_and_sizes[cur_grp_idx][off_sz_idx].push_back(sparse_offsets_and_sizes[off_sz_idx][baig_idx]);
                            }
                            treated_sparse = true;
                        }
                    }
                    if(!treated_sparse)
                    {
                        //Handle dense offsets

                        //Offsets of direct tensors must be relative to the beginning of the batch,
                        //but only if that tensor is batched over this index
                        //This is taken care of in sparsity_fuser for sparse offsets
                        idx_list direct_tensors = sf.get_direct_tensors();
                        size_t base_offset;
                        idx_list::iterator direct_pos = find(direct_tensors.begin(),direct_tensors.end(),bispace_idx);
                        if((direct_pos != direct_tensors.end()) && (batches.find(loop_idx) != batches.end()))
                        {
                            //Edge case for empty batch
                            if(loop_block_list.size() == 0)
                            {
                                base_offset = 0;
                            }
                            else
                            {
                                size_t min_idx = *(min_element(loop_block_list.begin(),loop_block_list.end()));
                                base_offset = subspace.get_block_abs_index(min_idx);
                            }
                        }
                        else
                        {
                            base_offset = 0;
                        }

                        for(size_t block_rel_idx = 0; block_rel_idx < loop_block_list.size(); ++block_rel_idx)
                        {
                            size_t offset = subspace.get_block_abs_index(loop_block_list[block_rel_idx]) - base_offset;
                            size_t size = subspace.get_block_size(loop_block_list[block_rel_idx]);
                            m_offsets_and_sizes[cur_grp_idx][block_rel_idx].push_back(off_dim_pair(offset,size));
                        }
                    }
                }
            }
        }
    }
    /*std::cout << "GROUPER FUNCTION SCOPE TIME: " << read_timer<double>() - seconds << "\n";*/
}


vector<idx_pair_list> sparse_loop_grouper::get_bispaces_and_index_groups() const
{
    return m_bispaces_and_index_groups;
}

vector<idx_pair_list> sparse_loop_grouper::get_bispaces_and_subspaces() const
{
    return m_bispaces_and_subspaces;
}

vector<vector<dim_list> > sparse_loop_grouper::get_block_dims() const
{
    return m_block_dims;
}

vector< vector<off_dim_pair_list> > sparse_loop_grouper::get_offsets_and_sizes() const
{
    return m_offsets_and_sizes;
}

vector<idx_list> sparse_loop_grouper::get_loops_for_groups() const
{
    return m_loops_for_groups;
}

} // namespace libtensor
