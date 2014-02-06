#ifndef CONTRACT_H
#define CONTRACT_H

#include "gen_labeled_btensor.h" 
#include "batch_provider_factory.h"
#include "block_contract2_kernel.h"

namespace libtensor {

template<typename T>
class contract2_batch_provider : public batch_provider<T>
{
private:
    static const char *k_clazz; //!< Class name
    block_contract2_kernel<T> m_bc2k;

    virtual void init(const std::vector<block_loop>& loops,
                      const idx_list& direct_tensors,
                      const std::vector<sparse_bispace_any_order>& truncated_bispaces,
                      const std::vector<T*>& ptrs,
                      const std::map<size_t,idx_pair>& loop_batches) 
    { 
        //Need to make sure C is zeroed out before contraction
        memset(ptrs[0],0,truncated_bispaces[0].get_nnz()*sizeof(T));
    };

    virtual void run_impl(const std::vector<block_loop>& loops,
                          const idx_list& direct_tensors,
                          const std::vector<sparse_bispace_any_order>& truncated_bispaces,
                          const std::vector<T*>& ptrs,
                          const std::map<size_t,idx_pair>& loop_batches)
    {

        sparse_loop_list sll(loops,direct_tensors);
        sll.run(m_bc2k,ptrs,loop_batches);
    }
public:
    contract2_batch_provider(const std::vector<block_loop>& loops,
                             const std::vector<size_t>& direct_tensors,
                             const std::vector<batch_provider<T>*>& batch_providers,
                             const std::vector<T*>& ptrs,
                             size_t mem_avail) : batch_provider<T>(loops,direct_tensors,batch_providers,ptrs,mem_avail),m_bc2k(sparse_loop_list(loops)) {}

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
