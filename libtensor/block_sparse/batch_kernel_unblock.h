#ifndef BATCH_KERNEL_UNBLOCK_H
#define BATCH_KERNEL_UNBLOCK_H

#include "batch_kernel.h"
#include "subspace_iterator.h"

namespace libtensor {

template<typename T>
class batch_kernel_unblock : public batch_kernel<T>
{
private:
    sparse_bispace_any_order m_bispace;
    size_t m_subspace_idx;
    size_t m_inner_size;
public:
    static const char* k_clazz; //!< Class name

    batch_kernel_unblock(const sparse_bispace_any_order& A,size_t subspace_idx);

    virtual void generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches);
};

template<typename T>
batch_kernel_unblock<T>::batch_kernel_unblock(const sparse_bispace_any_order& A,size_t subspace_idx) : m_bispace(A),
                                                                                                       m_subspace_idx(subspace_idx)
{
    if(m_bispace.get_n_index_groups() != m_bispace.get_order())
    {
        throw bad_parameter(g_ns, k_clazz,"batch_kernel_unblock(...)",__FILE__, __LINE__,
            "Cannot handle sparse bispaces");
    }
#if 0
    m_inner_size = 1;
    for(size_t i = m_bispace.get_index_group_containing_subspace(subspace_idx)+1; i < m_bispace.get_n_index_groups(); ++i)
    {
        m_inner_size *= m_bispace.get_index_group_dim(i);
    }
#endif
}
    
template<typename T>
void batch_kernel_unblock<T>::generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches)
{
    if(batches.size() > 0)
    {
#ifdef LIBTENSOR_DEBUG
        size_t subspace_idx;
        bool problem = false;
        if(batches.size() != 2)
        {
            problem = true;
        }
        else
        {
            subspace_idx = batches.begin()->first.second;
            if(batches.find(idx_pair(0,subspace_idx))->second != batches.find(idx_pair(1,subspace_idx))->second)
            {
                problem = true;

            }
        }
        if(problem) 
        {
                throw bad_parameter(g_ns, k_clazz,"batch_kernel_unblock::generate_batch(...)",__FILE__, __LINE__,
                        "Invalid batch info");
        }
#endif
    }

    const sparse_bispace<1>& subspace = m_bispace[m_subspace_idx];
    idx_pair batch = batches.size() > 0 ? batches.begin()->second : idx_pair(0,subspace.get_n_blocks());
    size_t batch_offset = subspace.get_block_abs_index(batch.first);
    size_t batch_subspace_dim;
    if(batch.second == subspace.get_n_blocks()) 
    {
        batch_subspace_dim = subspace.get_dim() - batch_offset;
    }
    else 
    {
        batch_subspace_dim = subspace.get_block_abs_index(batch.second) - batch_offset; 
    }

    size_t dest_off_base = 0;
    size_t dest_off = 0;
    idx_list idx_stack(m_bispace.get_order(),0);
    if(batches.size() > 0) idx_stack[m_subspace_idx] = batch.first;
    idx_list end_idx_stack;
    for(size_t i = 0; i < m_bispace.get_order(); ++i) 
    {
        end_idx_stack.push_back(m_bispace[i].get_n_blocks());
    }
    //if(batches.size() > 0) end_idx_stack[m_subspace_idx] = batch.second;

    //Advance source to point where batch starts 
    size_t src_off = 0;
    subspace_iterator it(m_bispace,m_subspace_idx);
    while(it.get_block_index() != idx_stack[m_subspace_idx])
    {
        src_off += it.get_slice_size();
        ++it;
    }

    size_t dest_inner_size = 1;
    for(size_t i = m_subspace_idx+1; i < m_bispace.get_order(); ++i) 
    {
        dest_inner_size *= m_bispace[i].get_dim();
    }

    bool all_done = false;
    while(!all_done)
    {
        size_t outer_size = 1;
        size_t inner_size = 1;
        //Register all dimensions of current block
        for(size_t subspace_idx = 0; subspace_idx < m_bispace.get_order(); ++subspace_idx)
        {
            size_t block_idx = idx_stack[subspace_idx];
            if(subspace_idx < m_subspace_idx) outer_size *= m_bispace[subspace_idx].get_block_size(block_idx);
            if(subspace_idx > m_subspace_idx) inner_size *= m_bispace[subspace_idx].get_block_size(block_idx);
        }

        size_t block_idx = idx_stack[m_subspace_idx];
        size_t block_offset = subspace.get_block_abs_index(block_idx);
        size_t block_size = subspace.get_block_size(block_idx);
        if((batch.first <= block_idx) && (block_idx < batch.second))
        {
            for(size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx)
            {
                for(size_t element_idx = 0; element_idx < block_size; ++element_idx)  
                {
                    size_t element_dest_off = dest_off + (outer_idx * batch_subspace_dim + (block_offset+element_idx - batch_offset)) * dest_inner_size;
                    size_t element_src_off = src_off + (outer_idx*block_size + element_idx ) * inner_size;
                    memcpy(ptrs[0]+element_dest_off,ptrs[1]+element_src_off,inner_size*sizeof(T));
                }
            }
            dest_off += inner_size;
        }
        src_off += outer_size*block_size*inner_size; 

        //Advance iterator stack
        for(size_t j = 1; j <= idx_stack.size(); ++j)
        { 
            size_t i = idx_stack.size() - j;
            idx_stack[i]++;
            if(idx_stack[i] == end_idx_stack[i])
            {
                if(i == 0)
                {
                    all_done = true;
                    break;
                }
                idx_stack[i] = 0;
            }
            else
            {
                if(i <= m_subspace_idx) 
                {
                    if(i < m_subspace_idx)
                    {
                        dest_off_base += outer_size*batch_subspace_dim*dest_inner_size;
                    }
                    dest_off = dest_off_base;
                }
                break;
            }
        }
    }
}

template<typename T>
const char* batch_kernel_unblock<T>::k_clazz = "batch_kernel_unblock";

} // namespace libtensor

#endif /* BATCH_KERNEL_UNBLOCK_H */
