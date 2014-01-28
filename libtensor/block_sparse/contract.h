#ifndef CONTRACT_H
#define CONTRACT_H

#include "labeled_sparse_btensor.h" 
#include "lazy_eval_functor.h"
#include "block_contract2_kernel.h"

//TODO: REMOVE
#include <iostream>

namespace libtensor {

//TODO some design idiom for dodging invalidation of these references
//without drastically increasing cost from copies?!
//K: degree of contraction
//M: Order of A
//N: Order of B
template<size_t K,size_t M, size_t N,typename T=double>
class contract_eval_functor : public lazy_eval_functor<M+N-(2*K),T> {
public:
    static const char *k_clazz; //!< Class name
private:
    const letter_expr<K> m_le;
    const labeled_sparse_btensor<M,T>& m_A;
    const labeled_sparse_btensor<N,T>& m_B;
public:
    //Evalutates the contraction and puts the result in C
    void operator()(labeled_sparse_btensor<M+N-(2*K),T>& C) const;

    //Constructor
    contract_eval_functor(const letter_expr<K>& le,const labeled_sparse_btensor<M,T>& A,const labeled_sparse_btensor<N,T>& B) : m_le(le),m_A(A),m_B(B) {} 
};

template<size_t K,size_t M, size_t N,typename T>
const char *contract_eval_functor<K,M,N,T>::k_clazz = "contract_eval_functor<K,M,N,T>";

//TODO: index fusion going to have to happen here
template<size_t K,size_t M, size_t N,typename T>
void contract_eval_functor<K,M,N,T>::operator()(labeled_sparse_btensor<M+N-(2*K),T>& C) const
{
    //Need to make sure C is zeroed out before contraction
    memset((T*)C.get_data_ptr(),0,C.get_bispace().get_nnz()*sizeof(T));

    //Build the loops for the contraction
    //First do the uncontracted indices
    std::vector< sparse_bispace_any_order > bispaces(1,C.get_bispace());
    bispaces.push_back(m_A.get_bispace());
    bispaces.push_back(m_B.get_bispace());
    
    std::vector<block_loop> uncontracted_loops;
    for(size_t i = 0; i < M+N-(2*K); ++i)
    {
        const letter& a = C.letter_at(i);

        //Ensure that this index should actually be appearing on the LHS
        if(m_le.contains(a))
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "an index cannot be contracted and appear in the output");
        }
        else if(m_A.contains(a) && m_B.contains(a))
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "both tensors cannot contain an uncontracted index");
        }

    	block_loop bl(bispaces);
    	bl.set_subspace_looped(0,i);
        if(m_A.contains(a))
        {
        	bl.set_subspace_looped(1,m_A.index_of(a));
        }
        else if(m_B.contains(a))
        {
        	bl.set_subspace_looped(2,m_B.index_of(a));
        }
        else
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "an index appearing in the result must be present in one input tensor");
        }
        uncontracted_loops.push_back(bl);
    }

    //Now the contracted indices
    std::vector<block_loop> contracted_loops;
    for(size_t k = 0; k < K; ++k)
    {
        const letter& a = m_le.letter_at(k);
        if((!m_A.contains(a)) || (!m_B.contains(a)))
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "a contracted index must appear in all RHS tensors");
        }

        block_loop bl(bispaces);
        bl.set_subspace_looped(1,m_A.index_of(a));
        bl.set_subspace_looped(2,m_B.index_of(a));
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
        std::cout << "Contraction loops outside!\n";
        loops.insert(loops.end(),contracted_loops.begin(),contracted_loops.end());
        loops.insert(loops.end(),uncontracted_loops.begin(),uncontracted_loops.end());
    }
    else
    {
        loops.insert(loops.end(),uncontracted_loops.begin(),uncontracted_loops.end());
        loops.insert(loops.end(),contracted_loops.begin(),contracted_loops.end());
    }

    sparse_loop_list sll(loops);
    block_contract2_kernel<T> bc2k(sll);

    std::vector<T*> ptrs(1,(T*)C.get_data_ptr());
    ptrs.push_back((T*)m_A.get_data_ptr());
    ptrs.push_back((T*)m_B.get_data_ptr());
    sll.run(bc2k,ptrs);
}

template<size_t K,size_t M,size_t N,typename T>
contract_eval_functor<K,M,N,T> contract(letter_expr<K> le,labeled_sparse_btensor<M,T> A,labeled_sparse_btensor<N,T> B)
{
    return contract_eval_functor<K,M,N,T>(le,A,B);
}



//Special case for one index contractions
template<size_t M,size_t N,typename T>
contract_eval_functor<1,M,N,T> contract(const letter& a,const labeled_sparse_btensor<M,T>& A,const labeled_sparse_btensor<N,T>& B)
{
    return contract_eval_functor<1,M,N,T>(letter_expr<1>(a),A,B);
}


} // namespace libtensor


#endif /* CONTRACT_H */
