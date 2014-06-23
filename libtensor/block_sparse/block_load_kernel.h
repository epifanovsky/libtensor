/*
 * block_load_kernel_new.h
 *
 *  Created on: Nov 19, 2013
 *      Author: smanzer
 */

#ifndef BLOCK_LOAD_KERNEL_H_
#define BLOCK_LOAD_KERNEL_H_

#include "block_kernel_i.h"
#include "sparse_bispace.h"

namespace libtensor
{

template<typename T>
class block_load_kernel : public block_kernel_i<block_load_kernel<T>,T>
{
private:
    static const char* k_clazz; //!< Class name
    const T* m_data_ptr;
    const sparse_bispace_any_order m_bispace;
    std::vector<size_t> cur_block_indices;
    void _load(T* output_ptr,const T* input_ptr,const dim_list& output_dims,size_t level=0);
public:
    block_load_kernel(const sparse_bispace_any_order& bispace,T* data_ptr);
	void operator()(const std::vector<T*>& ptrs, const std::vector< dim_list >& dim_lists);
};

template<typename T>
const char* block_load_kernel<T>::k_clazz = "block_load_kernel<T>";

template<typename T>
block_load_kernel<T>::block_load_kernel(const sparse_bispace_any_order& bispace,T* data_ptr) : m_data_ptr(data_ptr), m_bispace(bispace)
{
	//Sparsity is unsupported for row major loading
    if(m_bispace.get_n_sparse_groups() > 0)
    {
        throw bad_parameter(g_ns, k_clazz,"block_load_kernel(...)",
                __FILE__, __LINE__, "row-major loading not supported for sparse tensors");
    }
    for(size_t i = 0; i < m_bispace.get_order(); ++i)
    {
        cur_block_indices.push_back(0);
    }
}

template<typename T>
void block_load_kernel<T>::_load(T* output_ptr,const T* input_ptr,const dim_list& output_dims,size_t level)
{
    //Base case
    if(level == (output_dims.size() - 1))
    {
        for(size_t i = 0; i < output_dims.back(); ++i)
        {
            *(output_ptr++) =  (*input_ptr++);
        }
    }
    else
    {
            size_t input_inner_size = 1;
            for(size_t input_inner_size_idx = level+1; input_inner_size_idx < output_dims.size(); ++input_inner_size_idx)
            {
                input_inner_size *= m_bispace[input_inner_size_idx].get_dim();
            }

            size_t output_inner_size = 1;
            for(size_t output_inner_size_idx = level+1; output_inner_size_idx < output_dims.size(); ++output_inner_size_idx)
            {
                output_inner_size *= output_dims[output_inner_size_idx];
            }

            for(size_t i = 0; i < output_dims[level]; ++i)
            {
                _load(output_ptr+i*output_inner_size,input_ptr+i*input_inner_size,output_dims,level+1);
            }
    }
}

//It is assumed that the blocks will be accessed in lexicographic order
template<typename T>
void block_load_kernel<T>::operator()(const std::vector<T*>& ptrs, const std::vector< dim_list >& dim_lists)
{

	if(ptrs.size() != 1 || ptrs.size() != dim_lists.size())
	{
        throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                __FILE__, __LINE__, "must pass exactly one pointer and dim_list");
	}
    size_t offset = 0; 
    for(size_t i = 0; i < cur_block_indices.size(); ++i)
    {
        size_t inner_size = 1;
        for(size_t inner_size_idx = i+1; inner_size_idx < m_bispace.get_order(); ++inner_size_idx)
        {
            //TODO: This needs to call some sparsity-aware function to determine this
            inner_size *= m_bispace[inner_size_idx].get_dim();
        }
        offset += m_bispace[i].get_block_abs_index(cur_block_indices[i])*inner_size;
    }
    _load(ptrs[0],m_data_ptr+offset,dim_lists[0]);

    //Move to the next block
    size_t cur_idx = m_bispace.get_order() - 1;
    while(true)
    {
        cur_block_indices[cur_idx] += 1;
        if(cur_block_indices[cur_idx] == m_bispace[cur_idx].get_n_blocks())
        {
            cur_block_indices[cur_idx] = 0;
            if(cur_idx == 0)
            {
                break;
            }
            else
            {
                cur_idx--;
            }
        }
        else
        {
            break;
        }
    }
}

} /* namespace libtensor */

#endif /* BLOCK_LOAD_KERNEL_H_ */
