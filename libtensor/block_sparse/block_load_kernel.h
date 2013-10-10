#ifndef BLOCK_LOAD_KERNEL_H
#define BLOCK_LOAD_KERNEL_H

#include <vector>
#include "sparse_bispace.h"

//TODO REMOVE

#include <iostream>

namespace libtensor {

template<typename T> 
class block_load_kernel : public block_kernel_i<1,0,T> {
public:
    static const char *k_clazz; //!< Class name
private:
    const T* m_data_ptr;
    const sparse_bispace_generic_i* m_bispace;
    std::vector<size_t> cur_block_indices;
    void _load(T* output_ptr,const T* input_ptr,const dim_list& output_dims,size_t level=0);
public:
    block_load_kernel(const sparse_bispace_generic_i& bispace,T* data_ptr);

    void operator()(const sequence<1, T*>& output_ptrs, 
                    const sequence<0, T*>& input_ptrs,
                    const sequence<1, dim_list>& output_dims,
                    const sequence<0, dim_list>& input_dims);
};

template<typename T>
const char *block_load_kernel<T>::k_clazz = "block_load_kernel<T>";

template<typename T>
block_load_kernel<T>::block_load_kernel(const sparse_bispace_generic_i& bispace,T* data_ptr) : m_data_ptr(data_ptr), m_bispace(bispace.clone())
{
    for(size_t i = 0; i < m_bispace->get_order(); ++i)
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
                input_inner_size *= (*m_bispace)[input_inner_size_idx].get_dim();
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
void block_load_kernel<T>::operator()(const sequence<1, T*>& output_ptrs, 
                                      const sequence<0, T*>& input_ptrs,
                                      const sequence<1, dim_list>& output_dims,
                                      const sequence<0, dim_list>& input_dims)
{

    size_t block_offset = m_bispace->get_block_offset_canonical(cur_block_indices);
    _load(output_ptrs[0],m_data_ptr+block_offset,output_dims[0]);

    //Move to the next block
    size_t cur_idx = m_bispace->get_order() - 1;
    while(true)
    {
        cur_block_indices[cur_idx] += 1;
        if(cur_block_indices[cur_idx] == (*m_bispace)[cur_idx].get_n_blocks())
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

} // namespace libtensor

#endif /* BLOCK_LOAD_KERNEL_H */
