#ifndef BLOCK_LOOP_H
#define BLOCK_LOOP_H


#include <vector>
#include <utility>
#include <limits>
#include "../core/sequence.h"
#include "sparse_bispace.h"
#include "block_kernel_i.h"

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
                   

namespace impl
{

template<size_t M,size_t N,typename T>
void _run_internal(const std::vector< block_loop<M,N,T> >& loop_list,
                   block_kernel_i<M,N,T>& kernel,
                   const sequence<M,T*>& output_ptrs,
                   const sequence<N,const T*>& input_ptrs,
                   const sequence<M,sparse_bispace_any_order>& output_bispaces,
                   const sequence<N,sparse_bispace_any_order>& input_bispaces,
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
                                      sequence<M,std::vector<size_t> >& output_block_dims,
                                      sequence<N,std::vector<size_t> >& input_block_dims,
                                      sequence<M,std::vector<size_t> >& output_block_indices,
                                      sequence<N,std::vector<size_t> >& input_block_indices,
                                      size_t loop_idx);
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
                   sequence<M,std::vector<size_t> >& output_block_dims,
                   sequence<N,std::vector<size_t> >& input_block_dims,
                   sequence<M,std::vector<size_t> >& output_block_indices,
                   sequence<N,std::vector<size_t> >& input_block_indices,
                   size_t loop_idx)
{
    const block_loop<M,N,T>& cur_loop = loop_list[loop_idx];
    const sparse_bispace<1>& cur_bispace = cur_loop.get_non_ignored_bispace(output_bispaces,input_bispaces);
            
    block_list block_idxs = cur_loop.get_sig_block_list(loop_idx,output_bispaces,input_bispaces,output_block_indices,input_block_indices);

    for(size_t i = 0; i < block_idxs.size(); ++i)
    {
        size_t block_idx = block_idxs[i];
        size_t block_size = cur_bispace.get_block_size(block_idx);
        
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
            _run_internal(loop_list,kernel,output_ptrs,input_ptrs,output_bispaces,input_bispaces,
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

    impl::_run_internal(loop_list,kernel,output_ptrs,input_ptrs,output_bispaces,input_bispaces,
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

} // namespace libtensor

#endif /* BLOCK_LOOP_H */
