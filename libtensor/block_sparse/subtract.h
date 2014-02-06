#ifndef SUBTRACT_H
#define SUBTRACT_H

#include "gen_labeled_btensor.h" 
#include "batch_provider_factory.h"
#include "block_subtract2_kernel.h"
#include "block_add2_kernel.h"
#include "../iface/expr_exception.h"

namespace libtensor {

template<typename T>
class subtract2_batch_provider : public batch_provider<T>
{
public:
    static const char *k_clazz; //!< Class name
private:
    sparse_loop_list m_add_sll;
    sparse_loop_list m_sub_sll;
    std::vector<T*> m_add_ptrs;
    std::vector<T*> m_sub_ptrs;
    std::vector<size_t> m_direct_tensors;
    std::vector<batch_provider<T>* > m_batch_providers;
public:
    subtract2_batch_provider(const std::vector<block_loop>& add_loops,
                             const std::vector<block_loop>& sub_loops,
                             const std::vector<size_t>& direct_tensors,
                             const std::vector<batch_provider<T>*>& batch_providers,
                             const std::vector<T*>& ptrs) : m_add_sll(add_loops),m_sub_sll(sub_loops),m_add_ptrs(3,ptrs[1]),m_sub_ptrs(3,ptrs[2]) {}

    virtual void get_batch(T* output_batch_ptr,const std::map<idx_pair,idx_pair>& output_batches = (std::map<idx_pair,idx_pair>()))
    {
        //TODO: This will break for a '-=' type operation
        memset(output_batch_ptr,0,m_add_sll.get_bispaces()[0].get_nnz()*sizeof(T));

        block_add2_kernel<T> ba2k;
        m_add_ptrs[0] = output_batch_ptr;
        m_add_ptrs[1] = output_batch_ptr;
        m_add_sll.run(ba2k,m_add_ptrs);
        block_subtract2_kernel<T> bs2k;
        m_sub_ptrs[0] = output_batch_ptr;
        m_sub_ptrs[1] = output_batch_ptr;
        m_sub_sll.run(bs2k,m_sub_ptrs);
    }

};

template<size_t N,typename T>
class subtract2_batch_provider_factory : public batch_provider_factory<N,T> {
public:
    static const char *k_clazz; //!< Class name
private:
    sparse_bispace<N> m_A_bispace;
    sparse_bispace<N> m_B_bispace;
    const letter_expr<N> m_A_letter_expr;
    const letter_expr<N> m_B_letter_expr;
    T* m_A_data_ptr;
    T* m_B_data_ptr;
    batch_provider<T>* m_A_batch_provider;
    batch_provider<T>* m_B_batch_provider;
public:
    //Constructor
    subtract2_batch_provider_factory(const gen_labeled_btensor<N,T>& A,const gen_labeled_btensor<N,T>& B) : m_A_letter_expr(A.get_letter_expr()),m_B_letter_expr(B.get_letter_expr()),
                                                                                                            m_A_bispace(A.get_bispace()),m_B_bispace(B.get_bispace())
    {
        m_A_data_ptr = (T*) A.get_data_ptr();
        m_B_data_ptr = (T*) B.get_data_ptr();
        m_A_batch_provider = A.get_batch_provider();
        m_B_batch_provider = B.get_batch_provider();
    }

    //Creates a batch provider that will produce a given batch of C 
    virtual batch_provider<T>* get_batch_provider(gen_labeled_btensor<N,T>& C) const 
    {
        letter_expr<N> C_le(C.get_letter_expr());
        sparse_bispace_any_order C_bispace = C.get_bispace();

        //To obtain correct results 
        std::vector<sparse_bispace_any_order> add_bispaces(2,C_bispace);
        add_bispaces.push_back(m_A_bispace);
        std::vector<sparse_bispace_any_order> sub_bispaces(2,C_bispace);  
        sub_bispaces.push_back(m_B_bispace);

        //We must use separate loops so that sparse fusion does not occur when subtracting
        //sparse from dense
        std::vector<block_loop> add_loops;
        std::vector<block_loop> sub_loops;
        for(size_t i = 0; i < N; ++i)
        {
            const letter& a = C_le.letter_at(i);
            if(!m_A_letter_expr.contains(a) || !m_B_letter_expr.contains(a))
            {
                throw expr_exception(g_ns, k_clazz,"get_batch_provider()(...)",__FILE__, __LINE__,
                        "Any letter in a subtraction expression must appear in all tensors!");
            }
            if(m_A_letter_expr.index_of(a) != i || m_B_letter_expr.index_of(a) != i)
            {
                throw expr_exception(g_ns, k_clazz,"get_batch_provider()(...)",__FILE__, __LINE__,
                        "Permutations are not supported for subtraction at this time");
            }

            block_loop bl_add(add_bispaces);
            bl_add.set_subspace_looped(0,i);
            bl_add.set_subspace_looped(1,i);
            bl_add.set_subspace_looped(2,i);
            add_loops.push_back(bl_add);

            block_loop bl_sub(sub_bispaces);
            bl_sub.set_subspace_looped(0,i);
            bl_sub.set_subspace_looped(1,i);
            bl_sub.set_subspace_looped(2,i);
            sub_loops.push_back(bl_sub);
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
        return new subtract2_batch_provider<T>(add_loops,sub_loops,direct_tensors,batch_providers,ptrs);
    };
};

template<size_t N,typename T>
const char* subtract2_batch_provider_factory<N,T>::k_clazz = "subtract2_batch_provider_factory<N,T>";

template<size_t N,typename T>
subtract2_batch_provider_factory<N,T> operator-(const gen_labeled_btensor<N,T>& A,const gen_labeled_btensor<N,T>& B)
{
    return subtract2_batch_provider_factory<N,T>(A,B);
}

} // namespace libtensor

#endif /* SUBTRACT_H */
