#ifndef LOOP_LIST_SPARSITY_DATA_H
#define LOOP_LIST_SPARSITY_DATA_H

#include <vector>
#include <algorithm>
#include <map>
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
    const block_list& get_sig_block_list(const block_list& cur_block_idxs,size_t loop_idx) const;


    template<size_t M,size_t N,typename T>
    loop_list_sparsity_data(const std::vector< block_loop<M,N,T> >& loop_list,
                                                const sequence<M, sparse_bispace_any_order>& output_bispaces,
                                                const sequence<N, sparse_bispace_any_order>& input_bispaces);

};

template<size_t M,size_t N,typename T>
std::vector<size_t> loop_list_sparsity_data::get_loops_accessing_tree(const std::vector< block_loop<M,N,T> >& loop_list,size_t bispace_idx,size_t tree_start_idx,size_t tree_end_idx)
{
    std::vector<size_t> loops_accessing_tree(tree_end_idx - tree_start_idx);
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
            loops_accessing_tree[cur_idx - tree_start_idx] = loop_idx; 
        }
    }
    return loops_accessing_tree;
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

#ifdef SFM_DEBUG
        std::cout << "cur_subspace_idx: " << cur_subspace_idx << "\n";
        std::cout << "group_offset: " << group_offset << "\n";
        std::cout << "cur_order: " << cur_order << "\n";
#endif

        //Is the current index contained in this tree?
        if((group_offset <= cur_subspace_idx) && (cur_subspace_idx < group_offset+cur_order))
        {
#ifdef SFM_DEBUG
            std::cout << "group_idx: " << group_idx << "\n"; 
            std::cout << "processed_trees size: " << processed_trees.size() << "\n";
            std::cout << "processed_trees:\n";
            for(size_t j = 0; j < processed_trees.size(); ++j)
            {
                std::cout << "\t" <<  processed_trees[j] << "\n";
            }
#endif

            //Did we process this tree already?
            std::vector<size_t>::iterator pt_it = std::find(processed_trees.begin(),processed_trees.end(),group_idx); 
            if(pt_it == processed_trees.end())
            {
                //Figure out which loops access the tree
                //Each vector entry contains the loop index that accesses that sub-index of the tree
                std::vector<size_t> loops_accessing_tree = get_loops_accessing_tree(loop_list,bispace_idx,group_offset,group_offset+cur_order);


                //Permute the tree such that the order of its indices corresponds to the order that they
                //are traversed in the loop list
                std::vector< std::pair<size_t,size_t> > perm_pairs(cur_order);
                for(size_t i = 0; i < cur_order; ++i)
                {
                    perm_pairs[i].first = loops_accessing_tree[i];
                    perm_pairs[i].second = i;
                }
                sort(perm_pairs.begin(),perm_pairs.end());

                //TODO: Can probably merge this sort, etc..., clean up this stuff
                //Used to find which indices to use in fusion
                std::map<size_t,size_t> loop_to_tree;
                for(size_t i = 0; i < cur_order; ++i)
                {
                    loop_to_tree.insert(perm_pairs[i]);
                }

                //Does another tree already contain this index? If so, we must fuse. Otherwise, register new tree
                if(m_is_sparse[loop_idx])
                {
#if SFM_DEBUG
                    std::cout << "lhs order: " <<  m_trees[m_inter_tree_indices[loop_idx]].get_order() << "\n";
                    std::cout << "rhs order: " <<  cur_tree.get_order() << "\n";
#endif

                    //TODO: This screening into separate function!!!!
                    //Determine what indices to fuse: the shared indices between these trees
                    size_t lhs_tree_idx = m_inter_tree_indices[loop_idx];
                    std::vector<size_t> lhs_indices_all = m_inds_in_trees[lhs_tree_idx];
                    std::vector<size_t> rhs_indices_fused;

                    //Keep only indices in the LHS tree that also appear in the RHS tree
                    std::vector<size_t> lhs_indices_fused;
                    for(size_t lhs_idx = 0; lhs_idx < lhs_indices_all.size(); ++lhs_idx)
                    {
                        if(loop_to_tree.find(lhs_indices_all[lhs_idx]) != loop_to_tree.end())
                        {
                            //Shift to tree-relative indices
                            lhs_indices_fused.push_back(lhs_idx);
                        }
                    }

                    //See what loops that touch the RHS tree are in common with the LHS tree
                    for(std::map<size_t,size_t>::const_iterator ltt_it = loop_to_tree.begin(); ltt_it != loop_to_tree.end(); ++ltt_it)
                    {
                        if(std::binary_search(lhs_indices_fused.begin(),lhs_indices_fused.end(),ltt_it->first))
                        {
                            rhs_indices_fused.push_back(ltt_it->second);
                        }
                        else
                        {
                            //This index is not redundant by fusion, so it now belongs to the fused result
                            m_inds_in_trees[lhs_tree_idx].push_back(ltt_it->first);
                        }
                    }

                    //Keep it sorted
                    sort(m_inds_in_trees[lhs_tree_idx].begin(),m_inds_in_trees[lhs_tree_idx].end());


                    //Any indices from the current tree that we haven't fused now belong to the fused tree
#ifdef SFM_DEBUG
                    std::cout << "\nlhs_indices_fused:\n";
                    for(size_t j = 0; j < lhs_indices_fused.size(); ++j)
                    {
                        std::cout << lhs_indices_fused[j] << "\n";
                    }
                    std::cout << "\nrhs_indices_fused:\n";
                    for(size_t j = 0; j < rhs_indices_fused.size(); ++j)
                    {
                        std::cout << rhs_indices_fused[j] << "\n";
                    }
#endif
                    sparse_block_tree_any_order& the_tree = m_trees[m_inter_tree_indices[loop_idx]];
                    the_tree = the_tree.fuse(cur_tree,lhs_indices_fused,rhs_indices_fused);


                    //TODO: Need to modify m_inds_in_trees to add any new indices that appear...

                    //TODO: Remove DEBUG
#if SFM_DEBUG
                    std::cout << "FUSION RESULT:\n";
                    std::cout << "order:" << the_tree.get_order() << "\n";
                    for(sparse_block_tree_any_order::iterator it = the_tree.begin(); it != the_tree.end(); ++it)
                    {
                        for(size_t j = 0; j < the_tree.get_order(); ++j)
                        {
                            std::cout << it.key()[j] << ",";
                        }
                        std::cout << "\n";
                    }
#endif
                }
                else
                {
                    //Record the loops that access this tree, in the order that the loops are run
                    std::vector<size_t> tree_to_loop_perm_vec(cur_order);
                    for(size_t i = 0; i < perm_pairs.size(); ++i)
                    {
                        tree_to_loop_perm_vec[i] = perm_pairs[i].second;
                    }
                    runtime_permutation perm(tree_to_loop_perm_vec);

#ifdef SFM_DEBUG
                    std::cout << "perm:\n"; 
                    for(size_t i  = 0; i < cur_order; ++i)
                    {
                        std::cout << perm[i] << ",";
                    }
                    std::cout << "\n";
#endif
                    m_inds_in_trees.push_back(std::vector<size_t>());
                    for(size_t i = 0; i < cur_order; ++i)
                    {
                        m_inds_in_trees.back().push_back(loops_accessing_tree[perm[i]]);
                    }


                    m_is_sparse[loop_idx] = true;
                    m_inter_tree_indices[loop_idx] = m_trees.size();
                    m_trees.push_back(cur_tree.permute(perm));

#ifdef SFM_DEBUG
                    std::cout << "XXXXXXXXXXXXXXXXXXXXXXXXX\n";
                    std::cout << "m_inds_in_trees:\n";
                    for(size_t i = 0; i < m_inds_in_trees.size(); ++i)
                    {
                        for(size_t j = 0; j < m_trees[i].get_order(); ++j)
                        {
                            std::cout << m_inds_in_trees[i][j] << ",";
                        }
                        std::cout << "\n";
                    }
#endif
                }
                processed_trees.push_back(group_idx);
                processed_tree_offsets.push_back(m_inter_tree_indices[loop_idx]);
                break;
            }
            else
            {
                //Get the tree list index where the processed tree was stored
                size_t pt_pos = distance(processed_trees.begin(),pt_it);
                size_t tree_idx = processed_tree_offsets[pt_pos];
                m_is_sparse[loop_idx] = true;
                m_inter_tree_indices[loop_idx] = tree_idx;
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
            if(cur_bispace.get_order() > 1 && !cur_loop.m_output_ignore[m])
            {
                size_t cur_subspace_idx = cur_loop.m_output_bispace_indices[m];
                extract_trees(loop_list,loop_idx,cur_bispace,m,cur_subspace_idx,processed_tree_sets[m],processed_tree_offset_groups[m]);
            }
        }
        for(size_t n = 0; n < N; ++n)
        {
            const sparse_bispace_any_order& cur_bispace = input_bispaces[n];
            //No sparsity in 1d bispaces
            if(cur_bispace.get_order() > 1 && !cur_loop.m_input_ignore[n])
            {
                size_t cur_subspace_idx = cur_loop.m_input_bispace_indices[n];
                extract_trees(loop_list,loop_idx,cur_bispace,M+n,cur_subspace_idx,processed_tree_sets[M+n],processed_tree_offset_groups[M+n]);
            }
        }
    }


#ifdef SFM_DEBUG
    std::cout << "################## FINAL DEBUG #################\n";
    std::cout << "------------\n";
    std::cout << "m_is_sparse:\n";
    for(size_t i = 0; i < m_is_sparse.size(); ++i)
    {
        std::cout << m_is_sparse[i] << "\n";
    }
    std::cout << "------------\n";
    std::cout << "m_inds_in_trees:\n";
    for(size_t i = 0; i < m_inds_in_trees.size(); ++i)
    {
        for(size_t j = 0; j < m_trees[i].get_order(); ++j)
        {
            std::cout << m_inds_in_trees[i][j] << ",";
        }
        std::cout << "\n";
    }
#endif
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
        //Return full block list
        return m_full_block_lists[loop_idx];
    }
}

} // namespace libtensor

#endif /* LOOP_LIST_SPARSITY_DATA_H */
