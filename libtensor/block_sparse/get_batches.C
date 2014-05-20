#include "sparse_bispace.h"
#include <map>

using std::vector; 
using std::pair; 
using std::make_pair; 
using std::map; 

//TODO: DEBUG REMOVE
#include <iostream>
using std::cout;

namespace libtensor {

vector<idx_pair> get_batches(const vector<sparse_bispace_any_order>& bispaces,
                             const std::vector<idx_pair>& batched_bispaces_subspaces,
                             size_t max_n_elem)
{
    //Assemble a list of iterators over all the subspaces over which we will be batching 
    typedef pair<sparse_block_tree_any_order::iterator,sparse_block_tree_any_order::iterator> sparse_iter_pair;
    typedef idx_pair dense_iter_pair;
    vector<sparse_iter_pair> sparse_iter_pairs;
    vector<dense_iter_pair> dense_iter_pairs;
    map<size_t,size_t> sparse_bispace_offsets;
    map<size_t,size_t> dense_bispace_offsets;
    vector<sparse_block_tree_any_order> permuted_trees;
    vector<size_t> scale_facs;

    for(size_t bs_idx = 0; bs_idx < batched_bispaces_subspaces.size(); ++bs_idx)
    {
        size_t bispace_idx = batched_bispaces_subspaces[bs_idx].first;
        size_t subspace_idx = batched_bispaces_subspaces[bs_idx].second;
        const sparse_bispace_any_order& bispace = bispaces[bispace_idx];
        size_t idx_grp = bispace.get_index_group_containing_subspace(subspace_idx);
        size_t idx_grp_offset = bispace.get_index_group_offset(idx_grp);

        //Are we in a sparse group?
        bool sparse = false;
        size_t target_sparse_grp;
        for(size_t sparse_grp = 0; sparse_grp < bispace.get_n_sparse_groups(); ++sparse_grp)
        {
            if(bispace.get_sparse_group_offset(sparse_grp) == idx_grp_offset)
            {
                target_sparse_grp = sparse_grp;
                sparse = true;
                break;
            }
        }

        if(sparse)
        {
            //Sparse index group
            sparse_block_tree_any_order orig_tree = bispace.get_sparse_group_tree(target_sparse_grp);
            size_t tree_subspace = subspace_idx - idx_grp_offset;

            //We must place the batched index at 0 to ensure that it is strictly increasing so that we can batch over it
            runtime_permutation perm(orig_tree.get_order());
            perm.permute(0,tree_subspace);
            sparse_bispace_offsets[bs_idx] = permuted_trees.size();
            permuted_trees.push_back(orig_tree.permute(perm));
        }
        else
        {
            //Dense index group
            dense_bispace_offsets[bs_idx] = dense_iter_pairs.size();
            dense_iter_pairs.push_back(dense_iter_pair(0,bispace[subspace_idx].get_n_blocks()));
        }

        //Determine the scale factor (~stride) for this bispace/subspace pair
        size_t scale_fac = 1;
        for(size_t outer_idx_grp = 0; outer_idx_grp < idx_grp; ++outer_idx_grp)
        {
            scale_fac *= bispace.get_index_group_dim(outer_idx_grp);
        }
        for(size_t inner_idx_grp = idx_grp+1; inner_idx_grp < bispace.get_n_index_groups(); ++inner_idx_grp)
        {
            scale_fac *= bispace.get_index_group_dim(inner_idx_grp);
        }
        scale_facs.push_back(scale_fac);
    }

    //We must initialize the sparse iterators once all trees are processed so that the vector of trees
    //won't be resized and invalidate the iterators
    for(size_t tree_idx = 0; tree_idx < permuted_trees.size(); ++tree_idx)
    {
        sparse_iter_pairs.push_back(sparse_iter_pair(permuted_trees[tree_idx].begin(),permuted_trees[tree_idx].end()));
    }

    vector<idx_pair> batches;
    bool done = false;
    vector<size_t> halted_stat(batched_bispaces_subspaces.size(),false);
    vector<size_t> cur_block_inds(batched_bispaces_subspaces.size(),0);
    vector<size_t> cur_block_subtotals(batched_bispaces_subspaces.size(),0);
    vector<size_t> n_elems(batched_bispaces_subspaces.size(),0);

    //Used to pull back sparse iterators when another bispace fills the batch before they do
    vector< map<size_t,sparse_block_tree_any_order::iterator> > sparse_block_iters(sparse_iter_pairs.size());
    for(int i = 0; i < sparse_block_iters.size(); ++i)
    {
        sparse_block_iters[i].insert(make_pair(0,sparse_iter_pairs[i].first));
    }

    size_t batch_start_idx = 0;
    size_t bispace_idx = batched_bispaces_subspaces[0].first;
    size_t subspace_idx = batched_bispaces_subspaces[0].second;
    sparse_bispace<1> subspace = bispaces[bispace_idx][subspace_idx];
    while(!done)
    {
        for(size_t bs_idx = 0; bs_idx < batched_bispaces_subspaces.size(); ++bs_idx) 
        {
            //Check if all iterators are halted - we can then finish this batch
            bool all_halted = true;
            for(size_t halt_idx = 0; halt_idx < halted_stat.size(); ++halt_idx)
            {
                if(!halted_stat[halt_idx])
                {
                    all_halted = false;
                    break;
                }
            }
            if(all_halted)
            {
                vector<size_t>::iterator least_advanced_pos = std::min_element(cur_block_inds.begin(),cur_block_inds.end());
                size_t batch_end_idx = *least_advanced_pos;
                batches.push_back(idx_pair(batch_start_idx,batch_end_idx));
                batch_start_idx = batch_end_idx;

                //Are we done?
                if(batch_start_idx == subspace.get_n_blocks())
                {
                    done = true;
                    break;
                }

                //Move all dense iterators back to the start of the new batch
                for(size_t dense_iter_idx = 0; dense_iter_idx < dense_iter_pairs.size(); ++dense_iter_idx)
                {
                    dense_iter_pairs[dense_iter_idx].first = batch_start_idx;
                    cur_block_inds[dense_bispace_offsets[dense_iter_idx]] = batch_start_idx;
                } 

                //Roll back sparse iterators to the appropriate position to start the new batch
                for(size_t sparse_iter_idx = 0; sparse_iter_idx < sparse_iter_pairs.size(); ++sparse_iter_idx)
                {
                    sparse_iter_pairs[sparse_iter_idx].first = sparse_block_iters[sparse_iter_idx].lower_bound(batch_start_idx)->second;
                    cur_block_inds[sparse_bispace_offsets[sparse_iter_idx]] = sparse_iter_pairs[sparse_iter_idx].first.key()[0];
                }

                //All batches start fresh now
                n_elems.assign(n_elems.size(),0);
                cur_block_subtotals.assign(cur_block_subtotals.size(),0);
                halted_stat.assign(halted_stat.size(),false);
            }
            else if(halted_stat[bs_idx])
            {
                continue;
            }


            size_t scale_fac = scale_facs[bs_idx];
            size_t& cur_block_idx = cur_block_inds[bs_idx];
            size_t this_block_contrib;
            bool at_end;
            if(sparse_bispace_offsets.find(bs_idx) != sparse_bispace_offsets.end())
            {
                //Sparse case
                size_t sparse_offset = sparse_bispace_offsets[bs_idx];
                sparse_iter_pair& s_it = sparse_iter_pairs[sparse_offset];
                at_end = (s_it.first == s_it.second);
                if(!at_end)
                {
                    this_block_contrib = (*s_it.first)[0].second*scale_fac;

                    size_t block_idx = s_it.first.key()[0];
                    if(block_idx != cur_block_idx)
                    {
                        n_elems[bs_idx] += cur_block_subtotals[bs_idx];
                        cur_block_subtotals[bs_idx] = 0;
                        cur_block_idx = block_idx;
                        sparse_block_iters[sparse_bispace_offsets[bs_idx]].insert(make_pair(block_idx,s_it.first));
                    }

                    ++s_it.first;
                }
            }
            else
            {
                //Dense case
                size_t dense_offset = dense_bispace_offsets[bs_idx];
                dense_iter_pair& d_it = dense_iter_pairs[dense_offset];
                at_end = (d_it.first == d_it.second);
                if(!at_end)
                {
                    cur_block_idx = d_it.first;
                    this_block_contrib = subspace.get_block_size(cur_block_idx)*scale_fac;
                    n_elems[bs_idx] += cur_block_subtotals[bs_idx];
                    cur_block_subtotals[bs_idx] = 0;
                    ++d_it.first;
                }
            }
            cur_block_subtotals[bs_idx] += this_block_contrib;

            if(at_end)
            {
                cur_block_idx = subspace.get_n_blocks();
                halted_stat[bs_idx] = true;
            }
            else
            {
                if(cur_block_subtotals[bs_idx] > max_n_elem)
                {
                    throw bad_parameter(g_ns,"sparse_bispace<N>","get_batches(...)",__FILE__,__LINE__,
                            "single block does not fit in batch"); 
                }

                if(n_elems[bs_idx] + cur_block_subtotals[bs_idx] > max_n_elem)
                {
                    halted_stat[bs_idx] = true;
                }
            }
        }
    }

    return batches;
}


} // namespace libtensor
