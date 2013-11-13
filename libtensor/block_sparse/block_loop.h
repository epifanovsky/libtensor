#ifndef BLOCK_LOOP_H
#define BLOCK_LOOP_H


#include <vector>
#include <utility>
#include <limits>
#include "../core/sequence.h"
#include "sparse_bispace.h"
#include "block_kernel_i.h"
#include "loop_list_sparsity_data.h"

//TODO REMOVE:
#include <iostream>

namespace libtensor {

typedef std::pair<size_t,size_t> tile_size_pair;


//Forward declarations for 'friend' statement
//We need these to avoid linker errors 
template<size_t M,size_t N,typename T = double>
class block_loop;

template<size_t M,size_t N,typename T>
void run_loop_list(const std::vector< block_loop<M,N,T> >& loop_list,
                   block_kernel_i<M,N,T>& kernel,
                   const sequence<M,T*>& output_ptrs,
                   const sequence<N,const T*>& input_ptrs,
                   const sequence<M,sparse_bispace_any_order>& output_bispaces,
                   const sequence<N,sparse_bispace_any_order>& input_bispaces);

class loop_list_sparsity_data;

namespace impl
{

template<size_t M,size_t N,typename T>
void _run_internal(const std::vector< block_loop<M,N,T> >& loop_list,
                   block_kernel_i<M,N,T>& kernel,
                   const sequence<M,T*>& output_ptrs,
                   const sequence<N,const T*>& input_ptrs,
                   const sequence<M,sparse_bispace_any_order>& output_bispaces,
                   const sequence<N,sparse_bispace_any_order>& input_bispaces,
                   const loop_list_sparsity_data& llsd,
                   std::vector<size_t>& cur_block_idxs,
                   sequence<M,std::vector<size_t> >& output_block_dims,
                   sequence<N,std::vector<size_t> >& input_block_dims,
                   sequence<M,std::vector<size_t> >& output_block_indices,
                   sequence<N,std::vector<size_t> >& input_block_indices,
                   size_t loop_idx = 0);
} // namespace libtensor::impl

template<size_t M,size_t N,typename T>
class block_loop
{
public:
    static const char *k_clazz; //!< Class name
private:
    sequence<M,size_t> m_output_bispace_indices; //!< Which index in each output tensor does this loop modify?
    sequence<N,size_t> m_input_bispace_indices; //!< Which index in each input tensor does this loop modify?
    sequence<M,bool> m_output_ignore; //!< Which output pointers are incremented by this loop?
    sequence<N,bool> m_input_ignore; //!< Which input pointers are incremented by this loop?

    //Validates that all of the bispaces touched by this loop are equivalent
    void validate_bispaces(const sequence<M,sparse_bispace_any_order>& output_bispaces,
                           const sequence<N,sparse_bispace_any_order>& input_bispaces) const;


    //TODO: Merge back with get_non_ignored_bispace
    //Returns the index of the first 1D bispace in a tensor not ignored by this loop
    //Bispaces are numbered starting with output bispaces, then input bispaces
    size_t get_non_ignored_bispace_idx() const;

    //Returns the first bispace from the supplied bispaces that is touched by this loop and not from a tensor that is
    //ignored
    const sparse_bispace<1>& get_non_ignored_bispace(const sequence<M,sparse_bispace_any_order>& output_bispaces,
                                                     const sequence<N,sparse_bispace_any_order>& input_bispaces) const;

    //Used to determine the list of block indices over which this loop will iterate
    block_list get_sig_block_list(const size_t loop_idx,
                                  const sequence<M,sparse_bispace_any_order>& output_bispaces,
                                  const sequence<N,sparse_bispace_any_order>& input_bispaces,
                                  const sequence<M,std::vector<size_t> >& output_block_indices,
                                  const sequence<N,std::vector<size_t> >& input_block_indices) const;
public:

    //Constructor 
    block_loop(const sequence<M,size_t>& output_bispace_indices,
               const sequence<N,size_t>& input_bispace_indices,
               const sequence<M,bool>& output_ignore,
               const sequence<N,bool>& input_ignore); 

    //We friend the loop runner functions for convenience
    friend void run_loop_list<>(const std::vector< block_loop<M,N,T> >& loop_list,
                                block_kernel_i<M,N,T>& kernel,
                                const sequence<M,T*>& output_ptrs,
                                const sequence<N,const T*>& input_ptrs,
                                const sequence<M,sparse_bispace_any_order>& output_bispaces,
                                const sequence<N,sparse_bispace_any_order>& input_bispaces);


    friend void impl::_run_internal<>(const std::vector< block_loop<M,N,T> >& loop_list,
                                      block_kernel_i<M,N,T>& kernel,
                                      const sequence<M,T*>& output_ptrs,
                                      const sequence<N,const T*>& input_ptrs,
                                      const sequence<M,sparse_bispace_any_order>& output_bispaces,
                                      const sequence<N,sparse_bispace_any_order>& input_bispaces,
                                      const loop_list_sparsity_data& llsd,
                                      std::vector<size_t>& cur_block_idxs,
                                      sequence<M,std::vector<size_t> >& output_block_dims,
                                      sequence<N,std::vector<size_t> >& input_block_dims,
                                      sequence<M,std::vector<size_t> >& output_block_indices,
                                      sequence<N,std::vector<size_t> >& input_block_indices,
                                      size_t loop_idx);

    friend class loop_list_sparsity_data;
};


template<size_t M,size_t N,typename T>
const char *block_loop<M,N,T>::k_clazz = "block_loop<M,N>";

//Constructor
template<size_t M,size_t N,typename T>
block_loop<M,N,T>::block_loop(const sequence<M,size_t>& output_bispace_indices,
							  const sequence<N,size_t>& input_bispace_indices,
							  const sequence<M,bool>& output_ignore,
							  const sequence<N,bool>& input_ignore) : m_output_bispace_indices(output_bispace_indices),
													  		          m_input_bispace_indices(input_bispace_indices),
        															  m_output_ignore(output_ignore),
        															  m_input_ignore(input_ignore)
{
    bool all_ignored = true;
    for(size_t m = 0; m < M; ++m)
    {
        if(!m_output_ignore[m])
        {
            all_ignored = false;
            break;
        }
    }
    if(all_ignored)
    {
        for(size_t n = 0; n < N; ++n)
        {
            if(!m_input_ignore[n])
            {
                all_ignored = false;
                break;
            }
        }
    }
    if(all_ignored)
    {
        throw bad_parameter(g_ns, k_clazz,"block_loop(...)",
                __FILE__, __LINE__, "Cannot ignore all tensors: loop will do nothing");
    }
}

template<size_t M,size_t N,typename T>
size_t block_loop<M,N,T>::get_non_ignored_bispace_idx() const
{
    //Find the first output tensor that is not ignored, otherwise choose an input tensor
    //Constructor ensures that they are not all ignored
    bool in_output = false;
    size_t first_idx;
    for(size_t i = 0; i < M; ++i)
    {
        if(!m_output_ignore[i])
        {
            first_idx = i;
            in_output = true;
            break;
        }
    }
    if(in_output)
    {
        return first_idx;
    }
    else
    {
        for(size_t i = 0; i < N; ++i)
        {
            if(!m_input_ignore[i])
            {
                first_idx = i;
                break;
            }
        }
        return first_idx + M;
    }
}

//Returns a reference to the first 1D bispace in a tensor not ignored by this loop
//Necessary to deal with ignore feature
template<size_t M,size_t N,typename T>
const sparse_bispace<1>& block_loop<M,N,T>::get_non_ignored_bispace(const sequence<M,sparse_bispace_any_order>& output_bispaces,
                                                                    const sequence<N,sparse_bispace_any_order>& input_bispaces) const
{
    size_t first_idx = get_non_ignored_bispace_idx();
    if(first_idx < M)
    {
        return output_bispaces[first_idx][m_output_bispace_indices[first_idx]];
    }
    else
    {
        return input_bispaces[first_idx][m_input_bispace_indices[first_idx]];
    }
}

template<size_t M,size_t N,typename T>
void block_loop<M,N,T>::validate_bispaces(const sequence<M,sparse_bispace_any_order>& output_bispaces,
                                          const sequence<N,sparse_bispace_any_order>& input_bispaces) const
{
    const sparse_bispace<1>& ref_bispace = get_non_ignored_bispace(output_bispaces,input_bispaces); 
    //TODO BREAK UP THESE IFS SOME ARE REDUNDANT NOW!!!!!
    if(M != 0)
    {
        for(size_t i = 1; i < M; ++i)
        {
            if(!m_output_ignore[i]) 
            {
                if(ref_bispace != output_bispaces[i][m_output_bispace_indices[i]])
                {
                    throw bad_parameter(g_ns, k_clazz,"validate_bispaces(...)",
                            __FILE__, __LINE__, "Incompatible bispaces specified");
                }
            }
        }
        if(N != 0)
        {
            for(size_t i = 0; i < N; ++i)
            {
                if(!m_input_ignore[i])
                {
                    if(ref_bispace != input_bispaces[i][m_input_bispace_indices[i]])
                    {
                        throw bad_parameter(g_ns, k_clazz,"validate_bispaces(...)",
                                __FILE__, __LINE__, "Incompatible bispaces specified");
                    }
                }
            }
        }
    }
    else if(N != 0)
    {
        for(size_t i = 0; i < N; ++i)
        {
            if(! m_input_ignore[i])
            {
                if(! (ref_bispace == input_bispaces[i][m_input_bispace_indices[i]]) )
                {
                    throw bad_parameter(g_ns, k_clazz,"run(...)",
                            __FILE__, __LINE__, "Incompatible bispaces specified");
                }
            }
        }
    }
}

//We always just choose the shortest list
//Bispace code handles all accounding for sparsity etc
//TODO: Can probably streamline this by making fewer copies, just returning the LENGTH of each list
template<size_t M,size_t N,typename T>
block_list block_loop<M,N,T>::get_sig_block_list(const size_t loop_idx,
                                                 const sequence<M,sparse_bispace_any_order>& output_bispaces,
                                                 const sequence<N,sparse_bispace_any_order>& input_bispaces,
                                                 const sequence<M,std::vector<size_t> >& output_block_indices,
                                                 const sequence<N,std::vector<size_t> >& input_block_indices) const
{

    size_t min_len = std::numeric_limits<size_t>::max();
    block_list shortest_list;
    for(size_t output_idx = 0; output_idx < M; ++output_idx)
    {
        if(!m_output_ignore[output_idx])
        {
            const std::vector<size_t>& outer_block_indices = output_block_indices[output_idx];
            size_t target_subspace = m_output_bispace_indices[output_idx];

            //Is enough of the key specified to get a meaningful answer?
            if(target_subspace > loop_idx)
            {
                continue;
            }
            block_list cur_list = output_bispaces[output_idx].get_sig_block_list(outer_block_indices,target_subspace);
            if(cur_list.size() < min_len)
            {
                min_len = cur_list.size();
                shortest_list = cur_list;
            }
        }
    }

    for(size_t input_idx = 0; input_idx < N; ++input_idx)
    {
        if(!m_input_ignore[input_idx])
        {
            const std::vector<size_t>& outer_block_indices = input_block_indices[input_idx];
            size_t target_subspace = m_input_bispace_indices[input_idx];

            //Is enough of the key specified to get a meaningful answer?
            if(target_subspace > loop_idx)
            {
                continue;
            }
            block_list cur_list = input_bispaces[input_idx].get_sig_block_list(outer_block_indices,target_subspace);
            if(cur_list.size() < min_len)
            {
                min_len = cur_list.size();
                shortest_list = cur_list;
            }
        }
    }

    return shortest_list;
}

namespace impl
{

//Called recursively to run a kernel            
//INTERNAL USE ONLY                             
template<size_t M,size_t N,typename T>
void _run_internal(const std::vector< block_loop<M,N,T> >& loop_list,
                   block_kernel_i<M,N,T>& kernel,
                   const sequence<M,T*>& output_ptrs,
                   const sequence<N,const T*>& input_ptrs,
                   const sequence<M,sparse_bispace_any_order>& output_bispaces,
                   const sequence<N,sparse_bispace_any_order>& input_bispaces,
                   const loop_list_sparsity_data& llsd,
                   std::vector<size_t>& cur_block_idxs,
                   sequence<M,std::vector<size_t> >& output_block_dims,
                   sequence<N,std::vector<size_t> >& input_block_dims,
                   sequence<M,std::vector<size_t> >& output_block_indices,
                   sequence<N,std::vector<size_t> >& input_block_indices,
                   size_t loop_idx)
{
    const block_loop<M,N,T>& cur_loop = loop_list[loop_idx];
    const sparse_bispace<1>& cur_bispace = cur_loop.get_non_ignored_bispace(output_bispaces,input_bispaces);
    block_list block_idxs = llsd.get_sig_block_list(cur_block_idxs,loop_idx); 

    for(size_t i = 0; i < block_idxs.size(); ++i)
    {
        size_t block_idx = block_idxs[i];
        size_t block_size = cur_bispace.get_block_size(block_idx);
        
        cur_block_idxs[loop_idx] = block_idx;

        //TODO: This will need to increment along block loop for SPARSITY
        //will NOT use abs index in that case, just increment it within this loop
        size_t block_offset = cur_bispace.get_block_abs_index(block_idx);

        for(size_t m = 0; m < M; ++m)
        {
            if(cur_loop.m_output_ignore[m])
            {
                continue;
            }
            size_t cur_bispace_idx = cur_loop.m_output_bispace_indices[m];
            output_block_dims[m][cur_bispace_idx] = block_size;
            output_block_indices[m][cur_bispace_idx] = block_idx;
        }
        for(size_t n = 0; n < N; ++n)
        {
            if(cur_loop.m_input_ignore[n])
            {
                continue;
            }
            size_t cur_bispace_idx = cur_loop.m_input_bispace_indices[n];
            input_block_dims[n][cur_bispace_idx] = block_size;
            input_block_indices[n][cur_bispace_idx] = block_idx;
        }

        //Base case - use kernel to process the block 
        if(loop_idx == (loop_list.size() - 1))
        {
            sequence<M,T*> output_block_ptrs(output_ptrs);
            sequence<N,const T*> input_block_ptrs(input_ptrs);

            //Locate the appropriate blocks
            //TODO: this can be wasteful when tensors don't depend on a particular index - optimize this...
            for(size_t m = 0; m < M; ++m)
            {
                output_block_ptrs[m] += output_bispaces[m].get_block_offset(output_block_indices[m]); 
            }
            for(size_t n = 0; n < N; ++n)
            {
                input_block_ptrs[n] += input_bispaces[n].get_block_offset(input_block_indices[n]);
            }

            kernel(output_block_ptrs,input_block_ptrs,output_block_dims,input_block_dims);
        }
        else
        {
            _run_internal(loop_list,kernel,output_ptrs,input_ptrs,output_bispaces,input_bispaces,llsd,cur_block_idxs,
                          output_block_dims,input_block_dims,output_block_indices,input_block_indices,loop_idx+1);
        }
    }
}

} // namespace libtensor::impl

template<size_t M,size_t N,typename T>
void run_loop_list(const std::vector< block_loop<M,N,T> >& loop_list,
                   block_kernel_i<M,N,T>& kernel,
                   const sequence<M,T*>& output_ptrs,
                   const sequence<N,const T*>& input_ptrs,
                   const sequence<M,sparse_bispace_any_order>& output_bispaces,
                   const sequence<N,sparse_bispace_any_order>& input_bispaces)
{
    //Validate that the specified bispaces are all compatible for every loop in the list
    for(size_t loop_idx = 0; loop_idx < loop_list.size(); ++loop_idx)
    {
        loop_list[loop_idx].validate_bispaces(output_bispaces,input_bispaces);
    }

    //Prepare data structures for holding the current block dimensions and absolute indices for each tensor
    sequence<M,std::vector<size_t> > output_block_dims;
    sequence<N,std::vector<size_t> > input_block_dims;
    sequence<M,std::vector<size_t> > output_block_indices;
    sequence<N,std::vector<size_t> > input_block_indices;

    for(size_t m = 0; m < M; ++m)
    {
        output_block_dims[m].resize(output_bispaces[m].get_order());
        output_block_indices[m].resize(output_bispaces[m].get_order());

    }  
    for(size_t n = 0; n < N; ++n)
    { 
        input_block_dims[n].resize(input_bispaces[n].get_order());
        input_block_indices[n].resize(input_bispaces[n].get_order());
    }

    //Fuse all sparse trees from different tensors that are coupled by shared indices
    loop_list_sparsity_data llsd(loop_list,output_bispaces,input_bispaces);
    std::vector<size_t> cur_block_idxs(loop_list.size());

    impl::_run_internal(loop_list,kernel,output_ptrs,input_ptrs,output_bispaces,input_bispaces,llsd,cur_block_idxs,
            output_block_dims,input_block_dims,output_block_indices,input_block_indices);
}

//Overload for single loop argument
template<size_t M,size_t N,typename T>
void run_loop_list(block_loop<M,N,T>& loop,
                   block_kernel_i<M,N,T>& kernel,
                   const sequence<M,T*>& output_ptrs,
                   const sequence<N,const T*>& input_ptrs,
                   const sequence<M,sparse_bispace_any_order>& output_bispaces,
                   const sequence<N,sparse_bispace_any_order>& input_bispaces)
{
    run_loop_list(std::vector< block_loop<M,N,T> >(1,loop),kernel,
                  output_ptrs,input_ptrs,output_bispaces,input_bispaces);
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

                    //Keep it sorted for get_sig_block_list convenience
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

                    std::cout << "loops_accessing_tree:";
                    for(size_t i = 0; i < loops_accessing_tree.size(); ++i)
                    {
                        std::cout << "\t" << loops_accessing_tree[i] << "\n";
                    }
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
        if(ref_idx < M)
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



    //Finally, even if it is involved in a tree, we never need to consider sparsity for the first loop index in any free
    for(size_t tree_idx = 0; tree_idx < m_trees.size(); ++tree_idx) 
    {
        const std::vector<size_t>& rel_inds = m_inds_in_trees[tree_idx];
        m_is_sparse[rel_inds[0]] = false;
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

    //TODO: DEBUG REMOVE
    std::cout << "-----------------------\nTREES:\n";
    for(size_t tree_idx = 0; tree_idx < m_trees.size(); ++tree_idx) 
    {
        std::cout << "XXXXXXXXXXXXX\ntree: " << tree_idx << "\n";
        for(sparse_block_tree_any_order::iterator it = m_trees[tree_idx].begin(); it != m_trees[tree_idx].end(); ++it)
        {
            for(size_t j = 0; j < 3; ++j)
            {
                std::cout << it.key()[j] << ",";
            }
            std::cout << "\n";
        }
    }
    std::cout << "-----------------------\n";
#endif
}

} // namespace libtensor

#endif /* BLOCK_LOOP_H */
