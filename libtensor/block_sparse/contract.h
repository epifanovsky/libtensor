#ifndef CONTRACT_H
#define CONTRACT_H

#include "gen_labeled_btensor.h" 
#include "batch_provider_factory.h"
#include "block_contract2_kernel.h"

namespace libtensor {

template<typename T>
class contract2_batch_provider : public batch_provider<T>
{
public:
    static const char *k_clazz; //!< Class name
private:
    //TODO: ugly, shouldn't have m_loops and m_sll
    std::vector<block_loop> m_loops;
    sparse_loop_list m_sll;
    std::vector<T*> m_ptrs;
    block_contract2_kernel<T> m_bc2k;
    std::vector<size_t> m_direct_tensors;
    std::vector<batch_provider<T>* > m_batch_providers;
    size_t m_mem_avail;
public:
    contract2_batch_provider(const std::vector<block_loop>& loops,const std::vector<size_t>& direct_tensors,const std::vector<batch_provider<T>*>& batch_providers,const std::vector<T*>& ptrs,size_t mem_avail = 0) : m_loops(loops),
                                                                                                                                                                                                                       m_sll(loops,direct_tensors),
                                                                                                                                                                                                                       m_ptrs(ptrs),m_bc2k(m_sll),
                                                                                                                                                                                                                       m_direct_tensors(direct_tensors),
                                                                                                                                                                                                                       m_mem_avail(mem_avail),
                                                                                                                                                                                                                       m_batch_providers(batch_providers) {}
    virtual void get_batch(T* output_batch_ptr,const std::map<idx_pair,idx_pair>& output_batches = (std::map<idx_pair,idx_pair>()))
    {
        //Client code cannot assume a particular loop ordering, so it must specify the output batch
        //by specifying what bispace/subspace to truncate .
        //We now batch the loop that touches this bispace/subspace appropriately
        std::map<size_t,idx_pair> loop_batches;
        std::vector<sparse_bispace_any_order> bispaces = m_sll.get_bispaces();
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
                        bispaces[bispace_idx].truncate_subspace(subspace_idx,bounds);
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


        //Do we need to allocate batch memory for direct tensors that are used as inputs?
        std::vector<idx_pair> batches;
        size_t batched_loop_idx;
        if(direct_tensors_to_alloc.size() > 0)
        {
            //We will partition our available memory equally among them - it's a hack but whatever
            size_t mem_per_tensor = m_mem_avail/direct_tensors_to_alloc.size();

            //Do we need to batch beyond what is enforced by the output tensor batching? 
            //TODO: This currently only works for batching over a single loop 
            if(bispaces[direct_tensors_to_alloc[0]].get_nnz()*sizeof(T) > mem_per_tensor)
            {

                size_t bispace_idx = direct_tensors_to_alloc[0];
                sparse_bispace_any_order& bispace = bispaces[bispace_idx];

                //Find another loop accessing this bispace that isn't currently batched
                //Prioritize choosing a loop that accesses the most direct tensors
                bool found = false;
                size_t max_n_direct_tensors_touched = 0;
                for(size_t loop_idx = 0; loop_idx < m_loops.size(); ++loop_idx)
                {
                    const block_loop& loop = m_loops[loop_idx];
                    if(!loop.is_bispace_ignored(bispace_idx))
                    {
                        if(loop_batches.find(batched_loop_idx) == loop_batches.end())
                        {
                            found = true;
                            size_t n_direct_tensors_touched = 0;    
                            for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_alloc.size(); ++direct_tensor_rel_idx)
                            {
                                if(!loop.is_bispace_ignored(direct_tensors_to_alloc[direct_tensor_rel_idx]))
                                {
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
                }
                if(!found)
                {
                    throw bad_parameter(g_ns, k_clazz,"get_batch(...)",__FILE__, __LINE__,
                            "all subspaces are fully batched and tensor still does not fit in memory");
                            
                }
                const block_loop& batched_loop = m_loops[batched_loop_idx];

                //TODO: must consider ALL bispaces
                //Break the bispace down into batches
                size_t batched_subspace_idx = batched_loop.get_subspace_looped(bispace_idx);
                batches = bispace.get_batches(batched_subspace_idx,mem_per_tensor/sizeof(T));

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
                }
            }
        }

        //Compute the batch size
        size_t output_batch_size = bispaces[0].get_nnz()*sizeof(T);

        //Need to make sure C is zeroed out before contraction
        memset(output_batch_ptr,0,output_batch_size);

        //Place output in the provided batch memory
        m_ptrs[0] = output_batch_ptr;

        //Loop over the input direct tensor batches 
        if(batches.size() > 0)
        {
            //Generate the necessary input batches
            const block_loop& batched_loop = m_loops[batched_loop_idx];
            for(size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx)
            {
                loop_batches[batched_loop_idx] = batches[batch_idx];
                for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_alloc.size(); ++direct_tensor_rel_idx)
                {
                    size_t bispace_idx = m_direct_tensors[direct_tensor_rel_idx];
                    size_t subspace_idx = batched_loop.get_subspace_looped(bispace_idx);
                    std::map<idx_pair,idx_pair> this_bispace_batch;
                    this_bispace_batch[idx_pair(0,subspace_idx)] = batches[batch_idx];
                    m_batch_providers[direct_tensor_rel_idx]->get_batch(m_ptrs[bispace_idx],this_bispace_batch);
                }
                m_sll.run(m_bc2k,m_ptrs,loop_batches);
            }
        }
        else
        {
            //The input direct tensors can be formed as one batch
            for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_alloc.size(); ++direct_tensor_rel_idx)
            {
                size_t bispace_idx = m_direct_tensors[direct_tensor_rel_idx];
                m_batch_providers[direct_tensor_rel_idx]->get_batch(m_ptrs[bispace_idx]);
            }
            m_sll.run(m_bc2k,m_ptrs,loop_batches);
        }

        //Delete the batch memory that we allocated for the direct tensor inputs
        for(size_t direct_tensor_rel_idx = 0; direct_tensor_rel_idx < direct_tensors_to_alloc.size(); ++direct_tensor_rel_idx)
        {
            delete m_ptrs[direct_tensors_to_alloc[direct_tensor_rel_idx]];
        }
    }
};

template<typename T>
const char* contract2_batch_provider<T>::k_clazz = "contract2_batch_provider<T>";

template<size_t K,size_t M, size_t N,typename T>
class contract2_batch_provider_factory : public batch_provider_factory<M+N-(2*K),T> {
public:
    static const char *k_clazz; //!< Class name
private:
    const letter_expr<K> m_le;
    const letter_expr<M> m_A_letter_expr;
    const letter_expr<N> m_B_letter_expr;
    sparse_bispace<M> m_A_bispace;
    sparse_bispace<N> m_B_bispace;
    T* m_A_data_ptr;
    T* m_B_data_ptr;
    batch_provider<T>* m_A_batch_provider;
    batch_provider<T>* m_B_batch_provider;
    size_t m_mem_avail;
public:
    //Constructor
    contract2_batch_provider_factory(const letter_expr<K>& le,const gen_labeled_btensor<M,T>& A,const gen_labeled_btensor<N,T>& B,size_t mem_avail) : m_le(le),
                                                                                                                                                      m_A_letter_expr(A.get_letter_expr()),m_B_letter_expr(B.get_letter_expr()),
                                                                                                                                                      m_A_bispace(A.get_bispace()),m_B_bispace(B.get_bispace()),
                                                                                                                                                      m_mem_avail(mem_avail)

    {
        m_A_data_ptr = (T*) A.get_data_ptr();
        m_B_data_ptr = (T*) B.get_data_ptr();
        m_A_batch_provider = A.get_batch_provider();
        m_B_batch_provider = B.get_batch_provider();
    }

    //Creates a batch provider that will produce a given batch of C 
    virtual batch_provider<T>* get_batch_provider(gen_labeled_btensor<M+N-(2*K),T>& C) const 
    {
        letter_expr<M+N-(2*K)> C_le(C.get_letter_expr());
        //Build the loops for the contraction
        //First do the uncontracted indices
        std::vector< sparse_bispace_any_order > bispaces(1,C.get_bispace());
        bispaces.push_back(m_A_bispace);
        bispaces.push_back(m_B_bispace);
        
        std::vector<block_loop> uncontracted_loops;
        for(size_t i = 0; i < M+N-(2*K); ++i)
        {
            const letter& a = C_le.letter_at(i);

            //Ensure that this index should actually be appearing on the LHS
            if(m_le.contains(a))
            {
                throw bad_parameter(g_ns, k_clazz,"get_batch_provider()(...)",
                        __FILE__, __LINE__, "an index cannot be contracted and appear in the output");
            }
            else if(m_A_letter_expr.contains(a) && m_B_letter_expr.contains(a))
            {
                throw bad_parameter(g_ns, k_clazz,"get_batch_provider()(...)",
                        __FILE__, __LINE__, "both tensors cannot contain an uncontracted index");
            }

            block_loop bl(bispaces);
            bl.set_subspace_looped(0,i);
            if(m_A_letter_expr.contains(a))
            {
                bl.set_subspace_looped(1,m_A_letter_expr.index_of(a));
            }
            else if(m_B_letter_expr.contains(a))
            {
                bl.set_subspace_looped(2,m_B_letter_expr.index_of(a));
            }
            else
            {
                throw bad_parameter(g_ns, k_clazz,"get_batch_provider()(...)",
                        __FILE__, __LINE__, "an index appearing in the result must be present in one input tensor");
            }
            uncontracted_loops.push_back(bl);
        }

        //Now the contracted indices
        std::vector<block_loop> contracted_loops;
        for(size_t k = 0; k < K; ++k)
        {
            const letter& a = m_le.letter_at(k);
            if((!m_A_letter_expr.contains(a)) || (!m_B_letter_expr.contains(a)))
            {
                throw bad_parameter(g_ns, k_clazz,"get_batch_provider()(...)",
                        __FILE__, __LINE__, "a contracted index must appear in all RHS tensors");
            }

            block_loop bl(bispaces);
            bl.set_subspace_looped(1,m_A_letter_expr.index_of(a));
            bl.set_subspace_looped(2,m_B_letter_expr.index_of(a));
            contracted_loops.push_back(bl);
        }

        //Figure out whether we should make the loops over the contracted or uncontracted indices
        //the outer loops based on a crude estimate of their combined size.
        //We wanted contracted indices as outer loops for dot-product like things
        //TODO: account for sparsity here
        size_t uncontracted_dim = 1;
        for(size_t loop_idx = 0; loop_idx < uncontracted_loops.size(); ++loop_idx)
        {
            const block_loop& loop = uncontracted_loops[loop_idx];
            for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
            {
                if(!loop.is_bispace_ignored(bispace_idx))
                {
                    size_t subspace_idx = loop.get_subspace_looped(bispace_idx);
                    uncontracted_dim *= bispaces[bispace_idx][subspace_idx].get_dim();
                }
            }
        }
        size_t contracted_dim = 1;
        for(size_t loop_idx = 0; loop_idx < contracted_loops.size(); ++loop_idx)
        {
            const block_loop& loop = contracted_loops[loop_idx];
            for(size_t bispace_idx = 0; bispace_idx < bispaces.size(); ++bispace_idx)
            {
                if(!loop.is_bispace_ignored(bispace_idx))
                {
                    size_t subspace_idx = loop.get_subspace_looped(bispace_idx);
                    contracted_dim *= bispaces[bispace_idx][subspace_idx].get_dim();
                }
            }
        }
        std::vector<block_loop> loops;
        //Fudge factor of 2 for writes being more expensive 
        if(contracted_dim > uncontracted_dim*2)
        {
            loops.insert(loops.end(),contracted_loops.begin(),contracted_loops.end());
            loops.insert(loops.end(),uncontracted_loops.begin(),uncontracted_loops.end());
        }
        else
        {
            loops.insert(loops.end(),uncontracted_loops.begin(),uncontracted_loops.end());
            loops.insert(loops.end(),contracted_loops.begin(),contracted_loops.end());
        }

        //Direct tensors?
        std::vector<size_t> direct_tensors;
        std::vector<batch_provider<T>* > batch_providers;
        if(C.get_data_ptr() == NULL)
        {
            direct_tensors.push_back(0);
        }
        if(m_A_data_ptr == NULL)
        {
            direct_tensors.push_back(1);
            batch_providers.push_back(m_A_batch_provider);
        }
        if(m_B_data_ptr == NULL)
        {
            direct_tensors.push_back(2);
            batch_providers.push_back(m_B_batch_provider);
        }

        //Empty entry will be filled in by output batch
        std::vector<T*> ptrs(1);
        ptrs.push_back(m_A_data_ptr);
        ptrs.push_back(m_B_data_ptr);
        return new contract2_batch_provider<T>(loops,direct_tensors,batch_providers,ptrs,m_mem_avail);
    };
};

template<size_t K,size_t M, size_t N,typename T>
const char* contract2_batch_provider_factory<K,M,N,T>::k_clazz = "contract2_batch_provider_factory<K,M,N,T>";

template<size_t K,size_t M,size_t N,typename T>
contract2_batch_provider_factory<K,M,N,T> contract(letter_expr<K> le,const gen_labeled_btensor<M,T>& A,const gen_labeled_btensor<N,T>& B,size_t mem_avail = 0)
{
    return contract2_batch_provider_factory<K,M,N,T>(le,A,B,mem_avail);
}

//Special case for one index contractions
template<size_t M,size_t N,typename T>
contract2_batch_provider_factory<1,M,N,T> contract(const letter& a,const gen_labeled_btensor<M,T>& A,const gen_labeled_btensor<N,T>& B,size_t mem_avail = 0)
{
    return contract2_batch_provider_factory<1,M,N,T>(letter_expr<1>(a),A,B,mem_avail);
}

} // namespace libtensor


#endif /* CONTRACT_H */
