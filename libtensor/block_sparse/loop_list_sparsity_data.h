#ifndef LOOP_LIST_SPARSITY_DATA_H
#define LOOP_LIST_SPARSITY_DATA_H

#include <vector>
#include <algorithm>
#include <map>
#include "runtime_permutation.h"
#include "sparse_block_tree.h"
#include "sparse_bispace.h"

//TODO REMOVE
#include <iostream>

namespace libtensor {

//Forward declaration for argument types
template<size_t M,size_t N,typename T>
class block_loop;

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
    std::vector<size_t> get_loops_accessing_tree(const std::vector< block_loop<M,N,T> >& loop_list,size_t bispace_idx,size_t tree_start_idx,size_t tree_end_idx);

    template<size_t M,size_t N,typename T>
    void extract_trees(const std::vector< block_loop<M,N,T> >& loop_list,
                                            size_t loop_idx,
                                            const sparse_bispace_any_order& cur_bispace,
                                            size_t bispace_idx,
                                            size_t cur_subspace_idx,
                                            std::vector<size_t>& processed_trees,
                                            std::vector<size_t>& processed_tree_offsets);
public:
    block_list get_sig_block_list(const block_list& cur_block_idxs,size_t loop_idx) const;


    template<size_t M,size_t N,typename T>
    loop_list_sparsity_data(const std::vector< block_loop<M,N,T> >& loop_list,
                                                const sequence<M, sparse_bispace_any_order>& output_bispaces,
                                                const sequence<N, sparse_bispace_any_order>& input_bispaces);

};

inline block_list loop_list_sparsity_data::get_sig_block_list(const block_list& cur_block_idxs,size_t loop_idx) const
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
            if(rel_inds[rel_idx] == loop_idx)
            {
                break;
            }
            sub_key.push_back(cur_block_idxs[rel_inds[rel_idx]]);
        }

        //If the sub key can't be found, there are no blocks associated with it
        try
        {
            const block_list& bl = cur_tree.get_sub_key_block_list(sub_key);
            return bl;
        }
        catch(bad_parameter&)
        {
            return block_list();
        }
    }
    else
    {
        //Return full block list
        return m_full_block_lists[loop_idx];
    }
}

} // namespace libtensor

#endif /* LOOP_LIST_SPARSITY_DATA_H */
