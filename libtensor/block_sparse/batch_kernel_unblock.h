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
    bool m_src_direct;
public:
    static const char* k_clazz; //!< Class name

    batch_kernel_unblock(const sparse_bispace_any_order& A,size_t subspace_idx,bool src_direct=false);

    virtual void generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches);
};

template<typename T>
batch_kernel_unblock<T>::batch_kernel_unblock(const sparse_bispace_any_order& A,size_t subspace_idx,bool src_direct) : m_bispace(A),
                                                                                                                       m_subspace_idx(subspace_idx),
                                                                                                                       m_src_direct(src_direct)
{
    if(m_bispace.get_n_index_groups() != m_bispace.get_order())
    {
        throw bad_parameter(g_ns, k_clazz,"batch_kernel_unblock(...)",__FILE__, __LINE__,
            "Cannot handle sparse bispaces");
    }
}
    
template<typename T>
void batch_kernel_unblock<T>::generate_batch(const std::vector<T*>& ptrs,const bispace_batch_map& batches)
{
#ifdef LIBTENSOR_DEBUG
    if(batches.size() == 0 && m_src_direct)
        throw bad_parameter(g_ns, k_clazz,"batch_kernel_unblock::generate_batch(...)",__FILE__, __LINE__,
                "Invalid batch info");
#endif
    size_t batched_subspace_idx;
    if(batches.size() > 0)
    {
#ifdef LIBTENSOR_DEBUG
        bool problem = false;
        if(batches.size() != 2)
        {
            problem = true;
        }
        else
        {
            batched_subspace_idx = batches.begin()->first.second;
            if(batches.find(idx_pair(0,batched_subspace_idx))->second != batches.find(idx_pair(1,batched_subspace_idx))->second)
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

    const sparse_bispace<1>& unblocked_subspace = m_bispace[m_subspace_idx];
    size_t unblocked_subspace_dim = unblocked_subspace.get_dim();

    size_t next_outer_inds_off = 0;
    size_t outer_inds_off = 0;
    size_t dest_off = 0;
    idx_list idx_stack(m_bispace.get_order(),0);
    idx_list end_idx_stack;
    for(size_t i = 0; i < m_bispace.get_order(); ++i) 
    {
        end_idx_stack.push_back(m_bispace[i].get_n_blocks());
    }


    sparse_bispace_any_order batch_bispace(m_bispace);
    if(batches.size() > 0) batch_bispace.truncate_subspace(batched_subspace_idx,batches.begin()->second);
    size_t dest_inner_size = 1;
    for(size_t i = m_subspace_idx+1; i < m_bispace.get_order(); ++i) 
    {
        dest_inner_size *= batch_bispace[i].get_dim();
    }
    
    //Adjust for batching
    idx_pair batch; 
    size_t batch_offset = 0; 
    if(batches.size() > 0)
    {
        const sparse_bispace<1>& batched_subspace = m_bispace[batched_subspace_idx];
        batch = batches.begin()->second;
        if(m_src_direct)
        {
            idx_stack[batched_subspace_idx] = batch.first; 
            end_idx_stack[batched_subspace_idx] = batch.second; 
        }
        batch_offset = batched_subspace.get_block_abs_index(batch.first);
        if(batched_subspace_idx == m_subspace_idx)
        {
            if(batch.second == batched_subspace.get_n_blocks()) 
            {
                unblocked_subspace_dim = batched_subspace.get_dim() - batch_offset;
            }
            else 
            {
                unblocked_subspace_dim = batched_subspace.get_block_abs_index(batch.second) - batch_offset; 
            }
        }
    }

    bool all_done = false;
    size_t src_off = 0;
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

        size_t batched_block_idx = batches.size() > 0 ? idx_stack[batched_subspace_idx] : 0;
        size_t unblocked_block_idx = idx_stack[m_subspace_idx];
        size_t unblocked_block_offset = unblocked_subspace.get_block_abs_index(unblocked_block_idx);
        size_t unblocked_block_size = unblocked_subspace.get_block_size(unblocked_block_idx);
        if(batches.size() == 0 || ((batch.first <= batched_block_idx) && (batched_block_idx < batch.second)))
        {
            for(size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx)
            {
                for(size_t element_idx = 0; element_idx < unblocked_block_size; ++element_idx)
                {
                    size_t offset_in_idx_batch = unblocked_block_offset+ element_idx;
                    if(batched_subspace_idx == m_subspace_idx) offset_in_idx_batch -= batch_offset;
                    size_t element_dest_off = dest_off + (outer_idx * unblocked_subspace_dim + (offset_in_idx_batch)) * dest_inner_size;
                    size_t element_src_off = src_off + (outer_idx*unblocked_block_size + element_idx ) * inner_size;
                    memcpy(ptrs[0]+element_dest_off,ptrs[1]+element_src_off,inner_size*sizeof(T));
                }
            }
            dest_off += inner_size;
            next_outer_inds_off += outer_size*unblocked_block_size*inner_size;
        }
        src_off += outer_size*unblocked_block_size*inner_size; 

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
                idx_stack[i] = (m_src_direct && i == batched_subspace_idx) ? batch.first : 0;
            }
            else
            {
                if(i <= m_subspace_idx) 
                {
                    if(i < m_subspace_idx)
                    {
                        outer_inds_off = next_outer_inds_off;
                    }
                    dest_off = outer_inds_off;
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
