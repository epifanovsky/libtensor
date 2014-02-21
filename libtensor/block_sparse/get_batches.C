#include "sparse_bispace.h"
#include <map>

using std::vector; 
using std::pair; 
using std::map; 

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
            permuted_trees.push_back(orig_tree.permute(perm));
            sparse_bispace_offsets[bs_idx] = sparse_iter_pairs.size();
            sparse_iter_pairs.push_back(sparse_iter_pair(permuted_trees.back().begin(),permuted_trees.back().end()));
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

    vector<idx_pair> batches;
    bool done = false;
    vector<size_t> halted_stat(batched_bispaces_subspaces.size(),false);
    vector<size_t> cur_block_inds(batched_bispaces_subspaces.size(),0);
    vector<size_t> cur_block_subtotals(batched_bispaces_subspaces.size(),0);
    vector<size_t> start_inds(batched_bispaces_subspaces.size(),0);
    vector<size_t> n_elems(batched_bispaces_subspaces.size(),0);
    while(!done)
    {
        for(size_t bs_idx = 0; bs_idx < batched_bispaces_subspaces.size(); ++bs_idx) 
        {
            size_t bispace_idx = batched_bispaces_subspaces[bs_idx].first;
            size_t subspace_idx = batched_bispaces_subspaces[bs_idx].second;
            sparse_bispace<1> subspace = bispaces[bispace_idx][subspace_idx];

            size_t scale_fac = scale_facs[bs_idx];
            size_t& cur_block_idx = cur_block_inds[bs_idx];
            size_t& halted = halted_stat[bs_idx];
            if(sparse_bispace_offsets.find(bs_idx) != sparse_bispace_offsets.end())
            {
                //Sparse case
                size_t sparse_offset = sparse_bispace_offsets[bs_idx];
                sparse_iter_pair& s_it = sparse_iter_pairs[sparse_offset];
                if(s_it.first != s_it.second)
                {
                    size_t this_block_contrib = (*s_it.first)[0].second*scale_fac;

                    //TODO: This clause is conserved merge it with dense case!!!
                    if(this_block_contrib > max_n_elem)
                    {
                        throw bad_parameter(g_ns,"sparse_bispace<N>","get_batches(...)",__FILE__,__LINE__,
                                "single block does not fit in batch"); 
                            
                    }

                    size_t block_idx = s_it.first.key()[0];
                    if(block_idx != cur_block_idx)
                    {
                        n_elems[bs_idx] += cur_block_subtotals[bs_idx];
                        cur_block_subtotals[bs_idx] = 0;
                        cur_block_idx = block_idx;
                    }

                    cur_block_subtotals[bs_idx] += this_block_contrib;
                    //TODO: This clause is conserved merge it with dense case
                    if(n_elems[bs_idx] + cur_block_subtotals[bs_idx] > max_n_elem)
                    {
                        batches.push_back(idx_pair(start_inds[bs_idx],block_idx));
                        start_inds[bs_idx] = block_idx;
                        n_elems[bs_idx] = 0;
                    }
                    ++s_it.first;
                }
                else
                {
                    batches.push_back(idx_pair(start_inds[bs_idx],subspace.get_n_blocks()));
                    done = true;
                }
            }
            else
            {
                //Dense case
                size_t dense_offset = dense_bispace_offsets[bs_idx];
                dense_iter_pair& d_it = dense_iter_pairs[dense_offset];
                if(d_it.first != d_it.second)
                {
                    cur_block_idx = d_it.first;
                    /*if(halted)*/
                    /*{*/
                        /*//We don't have to be halted anymore if other iterators have caught up to us*/
                        /*if(cur_block_idx <= *std::min_element(cur_block_inds.begin(),cur_block_inds.end()))*/
                        /*{*/
                            /*halted = 0;*/
                        /*}*/
                    /*}*/
                    /*if(!halted)*/
                    /*{*/
                        size_t this_block_contrib = subspace.get_block_size(cur_block_idx)*scale_fac;
                        if(this_block_contrib > max_n_elem)
                        {
                            throw bad_parameter(g_ns,"sparse_bispace<N>","get_batches(...)",__FILE__,__LINE__,
                                    "single block does not fit in batch"); 
                        }

                        if(n_elems[bs_idx] + this_block_contrib > max_n_elem)
                        {
                            batches.push_back(idx_pair(start_inds[bs_idx],cur_block_idx));
                            start_inds[bs_idx] = cur_block_idx;
                            n_elems[bs_idx] = 0;

                            //TODO: Problems if other iterators have not advanced as far as this one?
                            //All other iterators must be set to start from the end of this batch
                            for(size_t other_bs_idx = 0; other_bs_idx < batched_bispaces_subspaces.size(); ++other_bs_idx)
                            {
                                if(sparse_bispace_offsets.find(other_bs_idx) != sparse_bispace_offsets.end())
                                {
                                    //Sparse case
                                }
                                else
                                {
                                    //Dense case
                                    dense_iter_pair& other_d_it = dense_iter_pairs[dense_bispace_offsets[other_bs_idx]];
                                    other_d_it.first = cur_block_idx;
                                }
                                n_elems[other_bs_idx] = 0;
                            }
                        }
                        n_elems[bs_idx] += this_block_contrib;
                        ++d_it.first;

                        //Halt and wait for other iterators to catch up
                        /*if(cur_block_idx > *std::min_element(cur_block_inds.begin(),cur_block_inds.end()))*/
                        /*{*/
                            /*halted = 1;*/
                        /*}*/
                    /*}*/
                }
                else
                {
                    batches.push_back(idx_pair(start_inds[bs_idx],subspace.get_n_blocks()));
                    done = true;
                }
            }
        }
    }

    return batches;
}


} // namespace libtensor
