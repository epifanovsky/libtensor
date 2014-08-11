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
    size_t dest_off_base = 0;
    size_t dest_off = 0;
    size_t src_off = 0;
    idx_list idx_stack(m_bispace.get_order(),0);
    idx_list end_idx_stack;
    for(size_t i = 0; i < m_bispace.get_order(); ++i) 
    {
        end_idx_stack.push_back(m_bispace[i].get_n_blocks());
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

        const sparse_bispace<1>& subspace = m_bispace[m_subspace_idx];
        size_t block_idx = idx_stack[m_subspace_idx];
        size_t block_offset = subspace.get_block_abs_index(block_idx);
        size_t block_size = subspace.get_block_size(block_idx);
        for(size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx)
        {
            for(size_t element_idx = 0; element_idx < block_size; ++element_idx)  
            {
                size_t element_dest_off = dest_off + (outer_idx*subspace.get_dim() + (block_offset+element_idx)) * dest_inner_size;
                size_t element_src_off = src_off + (outer_idx*block_size + element_idx ) * inner_size;
                memcpy(ptrs[0]+element_dest_off,ptrs[1]+element_src_off,inner_size*sizeof(T));
            }
        }
        dest_off += inner_size;
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
                        dest_off_base += subspace.get_dim()*dest_inner_size;
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
