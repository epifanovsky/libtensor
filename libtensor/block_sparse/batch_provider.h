#ifndef BATCH_PROVIDER_H
#define BATCH_PROVIDER_H

#include <map>
#include "block_loop.h"
#include "get_batches.h"

namespace libtensor {

//TODO: Should rewrite this using mixins/CRTP so that I don't need to INHERIT from the class too much baggage
//Constructor is too messy and huge...
template<typename T>
class batch_provider 
{
protected:
    static const char *k_clazz; //!< Class name
    std::vector<block_loop> m_loops;
    std::vector<T*> m_ptrs;
    std::vector<batch_provider<T>* > m_batch_providers;
    std::vector<size_t> m_direct_tensors;
    size_t m_mem_avail;
    idx_pair_list m_forced_batched_bs; 

    //Used - for example, to set the output tensor to zero prior to contraction
    virtual void init(const std::vector<block_loop>& loops,
                      const idx_list& direct_tensors,
                      const std::vector<sparse_bispace_any_order>& truncated_bispaces,
                      const std::vector<T*>& ptrs,
                      const std::map<size_t,idx_pair>& loop_batches) {};

    virtual void run_impl(const std::vector<block_loop>& loops,
                          const idx_list& direct_tensors,
                          const std::vector<sparse_bispace_any_order>& truncated_bispaces,
                          const std::vector<T*>& ptrs,
                          const std::map<size_t,idx_pair>& loop_batches) = 0;
public:
    batch_provider(const std::vector<block_loop>& loops,
                   const std::vector<size_t>& direct_tensors,
                   const std::vector<batch_provider<T>*>& batch_providers,
                   const std::vector<T*>& ptrs,
                   size_t mem_avail,
                   const idx_pair_list& forced_batched_bs = idx_pair_list()) : m_loops(loops),
                                       m_ptrs(ptrs),
                                       m_batch_providers(batch_providers),
                                       m_direct_tensors(direct_tensors),
                                       m_mem_avail(mem_avail),m_forced_batched_bs(forced_batched_bs) {}

    void set_mem_avail(size_t mem_avail) { m_mem_avail = mem_avail; }
    //Client code cannot assume a particular loop ordering, so it must specify the output batch
    //by specifying what bispace/subspace to truncate .
    void get_batch(T* output_batch_ptr,const std::map<idx_pair,idx_pair>& output_batches = (std::map<idx_pair,idx_pair>()),size_t mem_avail = 0);
    void set_forced_batched_bs(const idx_pair_list& forced_batched_bs) { m_forced_batched_bs = forced_batched_bs; }
    virtual ~batch_provider() {}
    virtual batch_provider<T>* clone() const = 0;
};

template<typename T>
void batch_provider<T>::get_batch(T* output_batch_ptr,const std::map<idx_pair,idx_pair>& output_batches,size_t mem_avail)
{
    //TODO: Hack for subtraction,permutation - need a better way
    if(mem_avail != 0)
    {
        m_mem_avail = mem_avail;
    }

    if(output_batches.size() > 1)
    {
        throw bad_parameter(g_ns, k_clazz,"get_batch(...)",__FILE__, __LINE__,
                "Can only batch over one subspace at a time at this time.");
    }
    //Do we need to allocate batch memory for direct tensors that are used as inputs?
    std::vector<idx_pair> batches;
    size_t batched_loop_idx = 0;

    //We now batch the loop that touches this bispace/subspace appropriately
    std::vector<sparse_bispace_any_order> bispaces = m_loops[0].get_bispaces();
    std::map<size_t,idx_pair> loop_batches;
    for(std::map<idx_pair,idx_pair>::const_iterator it = output_batches.begin(); it != output_batches.end(); ++it)
    {
        size_t bispace_idx = it->first.first;
        size_t subspace_idx = it->first.second;
        idx_pair bounds = it->second;
        for(size_t loop_idx = 0; loop_idx < m_loops.size(); ++loop_idx)
        {
            const block_loop& loop = m_loops[loop_idx];
            if(!loop.is_bispace_ignored(bispace_idx))
            {
                if(subspace_idx == loop.get_subspace_looped(bispace_idx))
                {
                    loop_batches[loop_idx] = bounds;

                    //Batch the whole loop
                    for(size_t cur_bispace_idx = 0; cur_bispace_idx < bispaces.size(); ++cur_bispace_idx)
                    {
                        if(!loop.is_bispace_ignored(cur_bispace_idx))
                        {
                            bispaces[cur_bispace_idx].truncate_subspace(loop.get_subspace_looped(cur_bispace_idx),bounds);
                        }
                    }
                    batched_loop_idx = loop_idx; 
                    batches.push_back(bounds);
                    break;
                }
            }
        }
    }

    //Find the direct tensors that don't already have memory allocated for them
    //TODO: Need some recursive implementation of this to be truly rigorous
    std::vector<size_t> direct_tensors_to_alloc;
    for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < m_direct_tensors.size(); ++direct_tensor_rel_idx)
    {
        size_t direct_tensor_idx = m_direct_tensors[direct_tensor_rel_idx];
        //It is an input direct tensor - we need to figure out how much batch memory to allocate for it
        if(direct_tensor_idx != 0)
        {
            direct_tensors_to_alloc.push_back(direct_tensor_idx);
        }
    }

    //See if we need to deal with forced batching considerations
    for(size_t loop_idx = 0; loop_idx < m_loops.size(); ++loop_idx)
    {
        const block_loop& loop = m_loops[loop_idx];
        for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
        {
            if(!loop.is_bispace_ignored(bispace_idx))
            {
                //Prioritize finding the loops corresponding to the bispace/subspace we are supposed to
                //batch
                size_t subspace_idx = loop.get_subspace_looped(bispace_idx);
                if(find(m_forced_batched_bs.begin(),m_forced_batched_bs.end(),idx_pair(bispace_idx,subspace_idx)) != m_forced_batched_bs.end())
                {
                    batched_loop_idx = loop_idx;
                    break;
                }
            }
        }
    }


    std::vector<size_t> direct_tensors_to_free;
    if(direct_tensors_to_alloc.size() > 0)
    {
        //We will partition our available memory equally among them - it's a hack but whatever
        size_t mem_per_tensor = m_mem_avail/direct_tensors_to_alloc.size();

        //Do we need to batch beyond what is enforced by the output tensor batching? 
        //TODO: This currently only works for batching over a single loop 
        //TODO: Make this check ALL TENSORS for fitting in batch
        if(bispaces[direct_tensors_to_alloc[0]].get_nnz()*sizeof(T) > mem_per_tensor)
        {
            //Prioritize choosing a loop that accesses the most direct tensors
            if(m_forced_batched_bs.size() == 0)
            {
                bool found = false;
                size_t max_n_direct_tensors_touched = 0;
                for(size_t loop_idx = 0; loop_idx < m_loops.size(); ++loop_idx)
                {
                    const block_loop& loop = m_loops[loop_idx];
                    //Can't use a loop that is already batched over
                    if(loop_batches.find(loop_idx) == loop_batches.end())
                    {
                        size_t n_direct_tensors_touched = 0;    
                        for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_alloc.size(); ++direct_tensor_rel_idx)
                        {
                            size_t bispace_idx = direct_tensors_to_alloc[direct_tensor_rel_idx];
                            if(!loop.is_bispace_ignored(bispace_idx))
                            {
                                found = true;
                                ++n_direct_tensors_touched;
                            }
                        }
                        if(n_direct_tensors_touched > max_n_direct_tensors_touched)
                        {
                            batched_loop_idx = loop_idx;
                            max_n_direct_tensors_touched = n_direct_tensors_touched;
                        }
                    }
                }
                if(!found)
                {
                    throw bad_parameter(g_ns, k_clazz,"get_batch(...)",__FILE__, __LINE__,
                            "all subspaces are fully batched and tensor still does not fit in memory");
                }
            }

            //Determine a batch structure that will make ALL batched tensors fit in memory
            const block_loop& batched_loop = m_loops[batched_loop_idx];
            std::vector<idx_pair> batched_bispaces_subspaces;
            for(size_t batched_bispace_idx = 0; batched_bispace_idx < bispaces.size(); ++batched_bispace_idx)
            {
                if(!batched_loop.is_bispace_ignored(batched_bispace_idx))
                {
                    if(binary_search(direct_tensors_to_alloc.begin(),direct_tensors_to_alloc.end(),batched_bispace_idx))
                    {
                        batched_bispaces_subspaces.push_back(idx_pair(batched_bispace_idx,batched_loop.get_subspace_looped(batched_bispace_idx)));
                    }
                }
            }
            batches = get_batches(bispaces,batched_bispaces_subspaces,mem_per_tensor/sizeof(T));

            //We will allocate memory large enough to hold the biggest batch for each bispace
            std::vector<size_t> max_batch_sizes(bispaces.size(),0);
            std::vector<size_t> max_batch_inds(bispaces.size(),0);
            for(size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
            {
                for(size_t cur_bispace_idx = 0; cur_bispace_idx < bispaces.size(); ++cur_bispace_idx)
                {
                    if(!batched_loop.is_bispace_ignored(cur_bispace_idx))
                    {
                        size_t cur_batched_subspace_idx = batched_loop.get_subspace_looped(cur_bispace_idx);
                        size_t batch_size = bispaces[cur_bispace_idx].get_batch_size(cur_batched_subspace_idx,batches[batch_idx]);
                        if(batch_size > max_batch_sizes[cur_bispace_idx])
                        {
                            max_batch_sizes[cur_bispace_idx] = batch_size;
                            max_batch_inds[cur_bispace_idx] = batch_idx;
                        }
                    }
                }
            }

            //Truncate all direct tensors to the size of the largest batch 
            //If any of the bispaces are still too big, code can't handle it right now
            for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_alloc.size(); ++direct_tensor_rel_idx) 
            {
                size_t cur_bispace_idx = direct_tensors_to_alloc[direct_tensor_rel_idx];
                if(!batched_loop.is_bispace_ignored(cur_bispace_idx))
                {
                    size_t subspace_idx = batched_loop.get_subspace_looped(cur_bispace_idx);
                    bispaces[cur_bispace_idx].truncate_subspace(subspace_idx,batches[max_batch_inds[cur_bispace_idx]]);
                }
            }
        }

        //Alloc the memory for input direct tensors
        for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_alloc.size(); ++direct_tensor_rel_idx) 
        {
            size_t cur_bispace_idx = direct_tensors_to_alloc[direct_tensor_rel_idx];
            if(bispaces[cur_bispace_idx].get_nnz()*sizeof(T) > mem_per_tensor)
            {
                throw bad_parameter(g_ns, k_clazz,"get_batch(...)",__FILE__, __LINE__, 
                    "after batching one loop a tensor still does not fit in memory");
                        
            }
            else
            {
                m_ptrs[cur_bispace_idx] = new T[bispaces[cur_bispace_idx].get_nnz()];
                direct_tensors_to_free.push_back(cur_bispace_idx);
            }
        }
    }

    //Compute the batch size
    size_t output_batch_size = bispaces[0].get_nnz()*sizeof(T);

    //Place output in the provided batch memory
    m_ptrs[0] = output_batch_ptr;

    init(m_loops,m_direct_tensors,bispaces,m_ptrs,loop_batches);

    //Loop over the input direct tensor batches 
    const block_loop& batched_loop = m_loops[batched_loop_idx];
    if(batches.size() > 0)
    {
        //Generate the necessary input batches
        for(size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
        {
            loop_batches[batched_loop_idx] = batches[batch_idx];

            for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_alloc.size(); ++direct_tensor_rel_idx)
            {
                size_t bispace_idx = direct_tensors_to_alloc[direct_tensor_rel_idx];
                size_t subspace_idx = batched_loop.get_subspace_looped(bispace_idx);
                std::map<idx_pair,idx_pair> this_bispace_batch;
                this_bispace_batch[idx_pair(0,subspace_idx)] = batches[batch_idx];
                m_batch_providers[direct_tensor_rel_idx]->set_mem_avail(m_mem_avail);
                if(!batched_loop.is_bispace_ignored(bispace_idx))
                {
                    m_batch_providers[direct_tensor_rel_idx]->set_forced_batched_bs(idx_pair_list(1,idx_pair(0,subspace_idx)));
                }
                m_batch_providers[direct_tensor_rel_idx]->get_batch(m_ptrs[bispace_idx],this_bispace_batch);
            }
            run_impl(m_loops,m_direct_tensors,bispaces,m_ptrs,loop_batches);
        }
    }
    else
    {
        //The input direct tensors can be formed as one batch
        for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_alloc.size(); ++direct_tensor_rel_idx)
        {
            size_t bispace_idx = direct_tensors_to_alloc[direct_tensor_rel_idx];
            size_t subspace_idx = batched_loop.get_subspace_looped(bispace_idx);

            m_batch_providers[direct_tensor_rel_idx]->set_mem_avail(m_mem_avail);
            if(!batched_loop.is_bispace_ignored(bispace_idx))
            {
                m_batch_providers[direct_tensor_rel_idx]->set_forced_batched_bs(idx_pair_list(1,idx_pair(0,subspace_idx)));
            }
            m_batch_providers[direct_tensor_rel_idx]->get_batch(m_ptrs[bispace_idx]);
        }
        run_impl(m_loops,m_direct_tensors,bispaces,m_ptrs,loop_batches);
    }



    //Delete the batch memory that we allocated for the direct tensor inputs
    for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_free.size(); ++direct_tensor_rel_idx)
    {
        delete [] m_ptrs[direct_tensors_to_free[direct_tensor_rel_idx]];
    }
}

template<typename T>
const char* batch_provider<T>::k_clazz = "batch_provider<T>";

} // namespace libtensor

#endif /* BATCH_PROVIDER_H */
