#ifndef LOOP_LIST_SPARSITY_DATA_H
#define LOOP_LIST_SPARSITY_DATA_H

#include <vector>
#include <algorithm>
#include "runtime_permutation.h"
#include "sparse_block_tree.h"
#include "block_loop.h"

//TODO REMOVE
#include <iostream>

namespace libtensor {

class loop_list_sparsity_data
{
private:
    std::vector<size_t> m_n_blocks;
    std::vector<bool> m_is_sparse;
    std::vector<size_t> m_inter_tree_indices;
    std::vector<size_t> m_intra_tree_idx;
    std::vector< sparse_block_tree_any_order > m_trees;
    std::vector< std::vector<size_t> > m_inds_in_trees;
    std::vector<size_t> m_cur_block_inds;
    std::vector< block_list > m_full_block_lists;

    //Helps constructor permute tree data into loop traversal index order
    template<size_t M,size_t N,typename T>
    runtime_permutation get_tree_permutation(const std::vector< block_loop<M,N,T> >& loop_list,size_t bispace_idx,size_t tree_start_idx,size_t tree_end_idx);

    template<size_t M,size_t N,typename T>
    void extract_trees(const std::vector< block_loop<M,N,T> >& loop_list,
                                            size_t loop_idx,
                                            const sparse_bispace_any_order& cur_bispace,
                                            size_t bispace_idx,
                                            size_t cur_subspace_idx,
                                            std::vector<size_t>& processed_trees,
                                            std::vector<size_t>& processed_tree_offsets);
public:
    const block_list& get_sig_block_list(const block_list& cur_block_idxs,size_t loop_idx) const;


    template<size_t M,size_t N,typename T>
    loop_list_sparsity_data(const std::vector< block_loop<M,N,T> >& loop_list,
                                                const sequence<M, sparse_bispace_any_order>& output_bispaces,
                                                const sequence<N, sparse_bispace_any_order>& input_bispaces);

};

template<size_t M,size_t N,typename T>
runtime_permutation loop_list_sparsity_data::get_tree_permutation(const std::vector< block_loop<M,N,T> >& loop_list,size_t bispace_idx,size_t tree_start_idx,size_t tree_end_idx)
{
    std::vector<size_t> perm_entries(tree_end_idx - tree_start_idx);
    size_t loop_ordered_idx = 0;
    for(size_t loop_idx = 0; loop_idx < loop_list.size(); ++loop_idx)
    {
        const block_loop<M,N,T>& cur_loop = loop_list[loop_idx];

        //Skip if this loop doesn't touch the bispace where the tree came from
        size_t cur_idx;
        if(bispace_idx < M)
        {
            if(cur_loop.m_output_ignore[bispace_idx])
            {
                continue;
            }
            cur_idx = cur_loop.m_output_bispace_indices[bispace_idx];
        }
        else
        {
            if(cur_loop.m_input_ignore[bispace_idx-M])
            {
                continue;
            }
            cur_idx = cur_loop.m_input_bispace_indices[bispace_idx-M];
        }

        //Does this loop touch an index in the tree?
        if((tree_start_idx <= cur_idx) && (cur_idx < tree_end_idx))
        {
            //Convert to tree-relative indices
            perm_entries[loop_ordered_idx] = cur_idx - tree_start_idx;
            ++loop_ordered_idx;
        }
    }
    return runtime_permutation(perm_entries);
}

//extracts the relevant trees from a set of bispaces, fusing them if necessary to produce a
//set of trees that accounts for the full set of couplings
template<size_t M,size_t N,typename T>
void loop_list_sparsity_data::extract_trees(const std::vector< block_loop<M,N,T> >& loop_list,
                                            size_t loop_idx,
                                            const sparse_bispace_any_order& cur_bispace,
                                            size_t bispace_idx,
                                            size_t cur_subspace_idx,
                                            std::vector<size_t>& processed_trees,
                                            std::vector<size_t>& processed_tree_offsets)
{
    //Determine if the current index is affected by any of the trees in this bispace?
    for(size_t group_idx = 0; group_idx < cur_bispace.get_n_sparse_groups(); ++group_idx)
    {
        const sparse_block_tree_any_order& cur_tree = cur_bispace.get_sparse_group_tree(group_idx);
        size_t group_offset = cur_bispace.get_sparse_group_offset(group_idx);
        size_t cur_order = cur_tree.get_order();

        //Is the current index contained in this tree?
        if((group_offset <= cur_subspace_idx) && (cur_subspace_idx < group_offset+cur_order))
        {
            //Did we process this tree already?
            std::vector<size_t>::iterator pt_it = std::find(processed_trees.begin(),processed_trees.end(),group_idx); 
            if(pt_it == processed_trees.end())
            {
                //Permute the tree such that the order of its indices corresponds to the order that they
                //are traversed in the loop list
                runtime_permutation perm = get_tree_permutation(loop_list,bispace_idx,group_offset,group_offset+cur_order);

                //Does another tree already contain this index? If so, we must fuse. Otherwise, register new tree
                if(m_is_sparse[loop_idx])
                {
                    //TODO: What fuse indices to pass?!!!!!!!
                    m_trees[m_inter_tree_indices[loop_idx]] = m_trees[m_inter_tree_indices[loop_idx]].fuse(cur_tree.permute(perm));
                }
                else
                {
                    m_is_sparse[loop_idx] = true;
                    processed_trees.push_back(group_idx);
                    processed_tree_offsets.push_back(m_trees.size());
                    m_trees.push_back(cur_tree.permute(perm));
                    m_inds_in_trees.push_back( std::vector<size_t>(1,loop_idx) );
                }
                break;
            }
            else
            {
                m_is_sparse[loop_idx] = true;
                //Get the tree list index where the processed tree was stored
                size_t pt_pos = distance(processed_trees.begin(),pt_it);
                size_t tree_idx = processed_tree_offsets[pt_pos];

                //Record this index as belonging to this tree
                //Don't double count from multiple tensors that feature this index
                if(m_inds_in_trees[tree_idx].back() != loop_idx)
                {
                    m_inds_in_trees[tree_idx].push_back(loop_idx);
                }
                break;
            }
        }
    }
}

template<size_t M,size_t N,typename T>
loop_list_sparsity_data::loop_list_sparsity_data(const std::vector< block_loop<M,N,T> >& loop_list,
                                            const sequence<M, sparse_bispace_any_order>& output_bispaces,
                                            const sequence<N, sparse_bispace_any_order>& input_bispaces) : m_n_blocks(loop_list.size()),
                                                                                                           m_is_sparse(loop_list.size(),false),
                                                                                                           m_inter_tree_indices(loop_list.size()),
                                                                                                           m_intra_tree_idx(loop_list.size()),
                                                                                                           m_cur_block_inds(loop_list.size())
{
    //Initialize block lists of all the blocks in each bispace for convenience 
    for(size_t loop_idx = 0; loop_idx < loop_list.size(); ++loop_idx)
    {
        const block_loop<M,N,T>& cur_loop = loop_list[loop_idx]; 
        size_t ref_idx = cur_loop.get_non_ignored_bispace_idx();

        block_list all_blocks;
        size_t n_blocks;
        if(ref_idx > M)
        {
            size_t subspace_idx = cur_loop.m_output_bispace_indices[ref_idx];
            n_blocks = output_bispaces[ref_idx][subspace_idx].get_n_blocks();
        }
        else
        {
            size_t subspace_idx = cur_loop.m_input_bispace_indices[ref_idx];
            n_blocks = input_bispaces[ref_idx][subspace_idx].get_n_blocks();
        }

        for(size_t i = 0; i < n_blocks; ++i)
        {
            all_blocks.push_back(i);
        }
        m_full_block_lists.push_back(all_blocks);
    }

    //Keep track of which trees have been processed already
    std::vector< std::vector<size_t> > processed_tree_sets(M+N);
    std::vector< std::vector<size_t> > processed_tree_offset_groups(M+N); 
    for(size_t loop_idx = 0; loop_idx < loop_list.size(); ++loop_idx)
    {
        const block_loop<M,N,T>& cur_loop = loop_list[loop_idx];
        for(size_t m = 0; m < M; ++m)
        {
            const sparse_bispace_any_order& cur_bispace = output_bispaces[m];
            //No sparsity in 1d bispaces
            if(cur_bispace.get_order() > 1)
            {
                size_t cur_subspace_idx = cur_loop.m_output_bispace_indices[m];
                extract_trees(loop_list,loop_idx,cur_bispace,m,cur_subspace_idx,processed_tree_sets[m],processed_tree_offset_groups[m]);
            }
        }
        for(size_t n = 0; n < N; ++n)
        {
            const sparse_bispace_any_order& cur_bispace = input_bispaces[n];
            //No sparsity in 1d bispaces
            if(cur_bispace.get_order() > 1)
            {
                size_t cur_subspace_idx = cur_loop.m_input_bispace_indices[n];
                extract_trees(loop_list,loop_idx,cur_bispace,M+n,cur_subspace_idx,processed_tree_sets[n],processed_tree_offset_groups[n]);
            }
        }
    }
}

//inline block_list range(size_t min,size_t max)
//{
    //block_list the_range; 
    //for(size_t i = min; i < max; ++i)
    //{
        //the_range.push_back(i);
    //}
    //return the_range;
//}

inline const block_list& loop_list_sparsity_data::get_sig_block_list(const block_list& cur_block_idxs,size_t loop_idx) const
{
    //Is there sparsity to apply to this index?
    if(m_is_sparse[loop_idx])
    {
        size_t tree_idx = m_inter_tree_indices[loop_idx];
        const sparse_block_tree_any_order& cur_tree = m_trees[tree_idx];

        //Assemble the current set of block indices that are relevant to this sparse tree
        block_list sub_key;
        const std::vector<size_t>& rel_inds = m_inds_in_trees[tree_idx];
        for(size_t rel_idx = 0; rel_idx < rel_inds.size(); ++rel_idx)
        {
            //Skip this one - we want the sub key
            if(rel_idx == loop_idx)
            {
                continue;
            }
            sub_key.push_back(cur_block_idxs[rel_inds[rel_idx]]);
        }
        
        return cur_tree.get_sub_key_block_list(sub_key);
    }
    else
    {
        std::cout << "\nWRONG!!!!!!!!!!!!!!!!!!!\n";
        //Return full block list
        return m_full_block_lists[loop_idx];
    }
}

} // namespace libtensor

#endif /* LOOP_LIST_SPARSITY_DATA_H */
