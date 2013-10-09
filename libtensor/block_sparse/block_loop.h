#ifndef BLOCK_LOOP_H
#define BLOCK_LOOP_H


#include <vector>
#include <utility>
#include "../core/sequence.h"
#include "sparse_bispace.h"
#include "block_kernel_i.h"

//TODO REMOVE:
#include <iostream>

namespace libtensor { 

typedef std::vector<size_t> block_list;
typedef std::pair<size_t,size_t> tile_size_pair;

block_list range(size_t min,size_t max)
{
    block_list the_range; 
    for(size_t i = min; i < max; ++i)
    {
        the_range.push_back(i);
    }
    return the_range;
}

template<size_t M,size_t N,typename T = double>
class block_loop
{
public:
    static const char *k_clazz; //!< Class name
private:
    block_loop<M,N,T>* m_inner_loop; //!< For nested loops
    block_kernel_i<M,N,T>* m_kernel;
    sequence<M,size_t> m_output_bispace_indices; //!< Which index in each output tensor does this loop modify?
    sequence<N,size_t> m_input_bispace_indices; //!< Which index in each input tensor does this loop modify?
    sequence<M,bool> m_output_ignore; //!< Which output pointers are incremented by this loop?
    sequence<N,bool> m_input_ignore; //!< Which input pointers are incremented by this loop?

    //Called recursively to run a kernel 
    void _run_internal(sequence<M,T*>& output_ptrs,
             sequence<N,T*>& input_ptrs,
             const sequence<M,sparse_bispace_generic_i*>& output_bispaces,
             const sequence<N,sparse_bispace_generic_i*>& input_bispaces,
             sequence<M,std::vector<size_t> >& output_block_dims,
             sequence<N,std::vector<size_t> >& input_block_dims,
             sequence<M,std::vector<size_t> >& output_block_offsets,
             sequence<N,std::vector<size_t> >& input_block_offsets);

    //Internal use only constructor called by nest
    block_loop(sequence<M,size_t>& output_bispace_indices,
               sequence<N,size_t>& input_bispace_indices,
               sequence<M,bool>& output_ignore,
               sequence<N,bool>& input_ignore,
               block_kernel_i<M,N,T>* kernel);

    //Validates that all of the bispaces touched by this loop are equivalent
    void validate_bispaces(const sequence<M,sparse_bispace_generic_i*>& output_bispaces,
                           const sequence<N,sparse_bispace_generic_i*>& input_bispaces);
public:

    //Constructor for the innermost loop - only the innermost loop should have a kernel
    block_loop(sequence<M,size_t>& output_bispace_indices,
               sequence<N,size_t>& input_bispace_indices,
               sequence<M,bool>& output_ignore,
               sequence<N,bool>& input_ignore,
               block_kernel_i<M,N,T>& kernel); 

    //Embed a nested loop into this loop, which will be run instead of the kernel
    void nest(sequence<M,size_t>& output_bispace_indices,
              sequence<N,size_t>& input_bispace_indices,
              sequence<M,bool>& output_ignore,
              sequence<N,bool>& input_ignore);
    

    //TODO: Overloaded version that takes a lambda functor for generating a given block instead of a pointer!!!
    void run(sequence<M,T*>& output_ptrs,
             sequence<N,T*>& input_ptrs,
             const sequence<M,sparse_bispace_generic_i*>& output_bispaces,
             const sequence<N,sparse_bispace_generic_i*>& input_bispaces);

};


template<size_t M,size_t N,typename T>
const char *block_loop<M,N,T>::k_clazz = "block_loop<M,N>";

template<size_t M,size_t N,typename T>
block_loop<M,N,T>::block_loop(sequence<M,size_t>& output_bispace_indices,
							  sequence<N,size_t>& input_bispace_indices,
							  sequence<M,bool>& output_ignore,
							  sequence<N,bool>& input_ignore,
							  block_kernel_i<M,N,T>& kernel) : m_output_bispace_indices(output_bispace_indices),
															   m_input_bispace_indices(input_bispace_indices),
															   m_output_ignore(input_ignore),
															   m_input_ignore(input_ignore)
{
    m_kernel = kernel.clone();
    m_inner_loop = NULL;
}

//Nest always puts the new loop inside the current innermost loop
template<size_t M,size_t N,typename T>
void block_loop<M,N,T>::nest(sequence<M,size_t>& output_bispace_indices,
							 sequence<N,size_t>& input_bispace_indices,
							 sequence<M,bool>& output_ignore,
							 sequence<N,bool>& input_ignore)

{
    if(m_inner_loop != NULL)
    {
        m_inner_loop->nest(output_bispace_indices,input_bispace_indices,output_ignore,input_ignore);
    }
    else
    {
        m_inner_loop = new block_loop<M,N,T>(output_bispace_indices,input_bispace_indices,output_ignore,input_ignore,m_kernel);
        m_kernel = NULL;
    }
}

//Internal use only! Called by nest
template<size_t M,size_t N,typename T>
block_loop<M,N,T>::block_loop(sequence<M,size_t>& output_bispace_indices,
							  sequence<N,size_t>& input_bispace_indices,
							  sequence<M,bool>& output_ignore,
							  sequence<N,bool>& input_ignore,
							  block_kernel_i<M,N,T>* kernel) : m_output_bispace_indices(output_bispace_indices),
															   m_input_bispace_indices(input_bispace_indices),
															   m_output_ignore(input_ignore),
															   m_input_ignore(input_ignore)
{
    m_kernel = kernel;
    m_inner_loop = NULL;
}

template<size_t M,size_t N,typename T>
void block_loop<M,N,T>::validate_bispaces(const sequence<M,sparse_bispace_generic_i*>& output_bispaces,
                                          const sequence<N,sparse_bispace_generic_i*>& input_bispaces)
{
    if(M != 0)
    {
        //TODO: A lot of copies here...this could bottleneck
        sparse_bispace<1> output_first = (*output_bispaces[0])[m_output_bispace_indices[0]];
        for(size_t i = 1; i < M; ++i)
        {
            if(! (output_first == (*output_bispaces[i])[m_output_bispace_indices[i]]) )
            {
                throw bad_parameter(g_ns, k_clazz,"run(...)",
                        __FILE__, __LINE__, "Incompatible bispaces specified");
            }
        }

        if(N != 0)
        {
            for(size_t i = 0; i < N; ++i)
            {
                if(! (output_first == (*input_bispaces[i])[m_input_bispace_indices[i]]) )
                {
                    throw bad_parameter(g_ns, k_clazz,"run(...)",
                            __FILE__, __LINE__, "Incompatible bispaces specified");
                }
            }
        }

    }
    else if(N != 0)
    {
        sparse_bispace<1> input_first = (*input_bispaces[0])[m_input_bispace_indices[0]];
        for(size_t i = 0; i < N; ++i)
        {
            if(! (input_first == (*input_bispaces[i])[m_input_bispace_indices[i]]) )
            {
                throw bad_parameter(g_ns, k_clazz,"run(...)",
                        __FILE__, __LINE__, "Incompatible bispaces specified");
            }
        }
    }

    if(m_inner_loop != NULL)
    {
        m_inner_loop->validate_bispaces(output_bispaces,input_bispaces);
    }
}

//Called recursively to run a kernel            
//INTERNAL USE ONLY                             
template<size_t M,size_t N,typename T>
void block_loop<M,N,T>::_run_internal(sequence<M,T*>& output_ptrs,
             sequence<N,T*>& input_ptrs,
             const sequence<M,sparse_bispace_generic_i*>& output_bispaces,
             const sequence<N,sparse_bispace_generic_i*>& input_bispaces,
             sequence<M,std::vector<size_t> >& output_block_dims,
             sequence<N,std::vector<size_t> >& input_block_dims,
             sequence<M,std::vector<size_t> >& output_block_offsets,
             sequence<N,std::vector<size_t> >& input_block_offsets)
{
	const sparse_bispace<1>& cur_bispace = (*output_bispaces[0])[m_output_bispace_indices[0]];
    block_list block_idxs = range(0,cur_bispace.get_n_blocks());

    for(size_t i = 0; i < block_idxs.size(); ++i)
    {
        size_t block_idx = block_idxs[i];
        size_t block_size = cur_bispace.get_block_size(block_idx);
        
        //TODO: This will need to increment along block loop for SPARSITY
        //will NOT use abs index in that case, just increment it within this loop
        size_t block_offset = cur_bispace.get_block_abs_index(block_idx);

        for(size_t m = 0; m < M; ++m)
        {
            size_t cur_bispace_idx = m_output_bispace_indices[m];
            output_block_dims[m][cur_bispace_idx] = block_size;
            output_block_offsets[m][cur_bispace_idx] = block_offset;
        }
        for(size_t n = 0; n < N; ++n)
        {
            size_t cur_bispace_idx = m_input_bispace_indices[n];
            input_block_dims[n][cur_bispace_idx] = block_size;
            input_block_offsets[n][cur_bispace_idx] = block_offset;
        }

        //Base case - use kernel to process the block 
        if(m_kernel != NULL)
        {
            sequence<M,T*> output_block_ptrs(output_ptrs);
            sequence<N,T*> input_block_ptrs(input_ptrs);

            //Locate the appropriate blocks
            for(size_t m = 0; m < M; ++m)
            {
                size_t offset = 0; 
                std::vector<size_t>& cur_output_block_dims = output_block_dims[m];
                size_t cur_order = cur_output_block_dims.size();
                for(size_t idx = 0; idx < cur_order; ++idx)
                {
                    //Compute outer size
                    size_t outer_size = 1;
                    for(size_t outer_size_idx = 0; outer_size_idx < idx; ++outer_size_idx)
                    {
                        outer_size *= cur_output_block_dims[outer_size_idx];
                    }

                    //TODO: Rewrite passing explicit block indices for handling sparsity
                    size_t inner_size = 1;
                    for(size_t inner_size_idx = idx+1; inner_size_idx < cur_order; ++inner_size_idx)
                    {
                        inner_size *= (*output_bispaces[m])[inner_size_idx].get_dim();
                    }

                    offset += outer_size * output_block_offsets[m][idx] * inner_size;
                }

                output_block_ptrs[m] += offset; 
            }
            for(size_t n = 0; n < N; ++n)
            {
                size_t offset = 0; 
                std::vector<size_t>& cur_input_block_dims = input_block_dims[n];
                size_t cur_order = cur_input_block_dims.size();
                for(size_t idx = 0; idx < cur_order; ++idx)
                {
                    //Compute outer size
                    size_t outer_size = 1;
                    for(size_t outer_size_idx = 0; outer_size_idx < idx; ++outer_size_idx)
                    {
                        outer_size *= cur_input_block_dims[outer_size_idx];
                    }

                    //TODO: Rewrite passing explicit block indices for handling sparsity
                    size_t inner_size = 1;
                    size_t cur_bispace_idx = m_input_bispace_indices[n];
                    for(size_t inner_size_idx = idx+1; inner_size_idx < cur_order; ++inner_size_idx)
                    {
                        inner_size *= (*input_bispaces[n])[inner_size_idx].get_dim();
                    }

                    offset += outer_size * input_block_offsets[n][idx] * inner_size;
                }
                input_block_ptrs[n] += offset;
            }

            (*m_kernel)(output_block_ptrs,input_block_ptrs,output_block_dims,input_block_dims);

        }
        else
        {
            m_inner_loop->_run_internal(output_ptrs,input_ptrs,output_bispaces,input_bispaces,
                                output_block_dims,input_block_dims,output_block_offsets,input_block_offsets);
        }
    }
}

template<size_t M,size_t N,typename T>
void block_loop<M,N,T>::run(sequence<M,T*>& output_ptrs,
         sequence<N,T*>& input_ptrs,
         const sequence<M,sparse_bispace_generic_i*>& output_bispaces,
         const sequence<N,sparse_bispace_generic_i*>& input_bispaces)
{
    //Validate that the specified bispaces are all compatible
    validate_bispaces(output_bispaces,input_bispaces);

    //Prepare data structures for holding the current block dimensions and absolute indices for each tensor
    sequence<M,std::vector<size_t> > output_block_dims;
    sequence<N,std::vector<size_t> > input_block_dims;
    sequence<M,std::vector<size_t> > output_block_offsets;
    sequence<N,std::vector<size_t> > input_block_offsets;

    for(size_t m = 0; m < M; ++m)
    {
        output_block_dims[m].resize(output_bispaces[m]->get_order());
        output_block_offsets[m].resize(output_bispaces[m]->get_order());

    }  
    for(size_t n = 0; n < N; ++n)
    { 
        input_block_dims[n].resize(input_bispaces[n]->get_order());
        input_block_offsets[n].resize(input_bispaces[n]->get_order());
    }

    _run_internal(output_ptrs,input_ptrs,output_bispaces,input_bispaces,
            output_block_dims,input_block_dims,output_block_offsets,input_block_offsets);
}

#if 0
template<size_t M,size_t N,typename T>
void block_loop<M,N,T>::run(sequence<M,T*>& output_ptrs,
                            sequence<N,T*>& input_ptrs,
                            sequence<M,std::vector<size_t> >& output_block_dims,
                            sequence<N,std::vector<size_t> >& input_block_dims,
                            sequence<M,size_t>& output_outer_strides,
                            sequence<N,size_t>& input_outer_strides)
{
    block_list block_idxs = range(0,m_bispace.get_n_blocks());

    sequence<M,T*> output_block_ptrs(output_ptrs);
    sequence<N,T*> input_block_ptrs(input_ptrs);

    for(size_t i = 0; i < block_idxs.size(); ++i)
    {
        size_t block_idx = block_idxs[i];
        size_t block_size = m_bispace.get_block_size(block_idx);

        for(size_t m = 0; m < M; ++m)
        {
            output_block_dims[m].push_back(block_size);
        }
        for(size_t n = 0; n < N; ++n)
        {
            input_block_dims[n].push_back(block_size);
        }
        //Base case - use kernel to process the block 
        if(m_kernel != NULL)
        {
            //Construct blocks for kernel
            sequence<M, block<T> > output_blocks;
            sequence<N, block<T> > input_blocks;
            for(size_t m = 0; m < M; ++m)
            {
                output_blocks[m] = block<T>(output_block_ptrs[m],output_block_dims[m]);
            }
            for(size_t n = 0; n < N; ++n)
            {
                input_blocks[n] = block<T>(input_block_ptrs[n],input_block_dims[n]);
            }

            (*m_kernel)(output_blocks,input_blocks);
        }
        else
        {
            //TODO: this will break with sparsity!!!
            //Inner tile size is now smaller 
            
            sequence<M, tile_size_pair > new_output_outer_strides(output_outer_strides);
            sequence<N, tile_size_pair > new_input_outer_strides(input_outer_strides);

            //Outer tile size is now bigger
            for(size_t m = 0; m < M; ++m)
            {
                new_output_outer_strides[m] *= block_size;
            }
            for(size_t n = 0; n < N; ++n)
            {
                new_input_outer_strides[n] *= block_size;
            }

            //Run the next loop
            m_next_loop->run(output_block_ptrs,input_block_ptrs,
                             output_block_dims,input_block_dims,
                             new_output_tile_sizes,new_input_tile_sizes);
        }

        //Move to the next block
        for(size_t m = 0; m < M; ++m)
        {
            output_block_dims[m].pop_back();
            output_block_ptrs[m] += output_tile_sizes[m].first * block_size * output_tile_sizes[m].second; 
        }
        for(size_t n = 0; n < N; ++n)
        {
            input_block_dims[n].pop_back();
            input_block_ptrs[n] += input_tile_sizes[n].first * block_size * input_tile_sizes[n].second; 
        }
    }
}
#endif

} // namespace libtensor

#endif /* BLOCK_LOOP_H */
