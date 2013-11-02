#ifndef CONTRACT_H
#define CONTRACT_H

#include "labeled_sparse_btensor.h" 
#include "lazy_eval_functor.h"

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
    std::vector< block_loop<1,2> > loop_list;
    std::vector< sequence<1,size_t> > output_indices_sets;
    std::vector< sequence<2,size_t> > input_indices_sets;
    std::vector< sequence<1,bool> > output_ignore_sets;
    std::vector< sequence<2,bool> > input_ignore_sets;
    for(size_t i = 0; i < M+N-(2*K); ++i)
    {
        sequence<1,size_t> output_bispace_indices(i);
        sequence<2,size_t> input_bispace_indices;
        sequence<1,bool> output_ignore(false);
        sequence<2,bool> input_ignore(false);

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

        size_t rhs_idx;
        if(m_A.contains(a))
        {
            input_bispace_indices[0] = m_A.index_of(a);  
            input_ignore[1] = true;
        }
        else if(m_B.contains(a))
        {
            input_bispace_indices[1] = m_B.index_of(a);
            input_ignore[0] = true;
        }
        else
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "an index appearing in the result must be present in one input tensor");
        }

        loop_list.push_back(block_loop<1,2>(output_bispace_indices,
                                            input_bispace_indices,
                                            output_ignore,
                                            input_ignore));
        output_indices_sets.push_back(output_bispace_indices);
        input_indices_sets.push_back(input_bispace_indices);
        output_ignore_sets.push_back(output_ignore);
        input_ignore_sets.push_back(input_ignore);
    }

    //Now the contracted indices
    for(size_t k = 0; k < K; ++k)
    {
        const letter& a = m_le.letter_at(k);
        if((!m_A.contains(a)) || (!m_B.contains(a)))
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "a contracted index must appear in all RHS tensors");
        }

        sequence<1,size_t> output_bispace_indices;
        sequence<2,size_t> input_bispace_indices;
        sequence<1,bool> output_ignore(true);
        sequence<2,bool> input_ignore(false);

        input_bispace_indices[0] = m_A.index_of(a);
        input_bispace_indices[1] = m_B.index_of(a);

        loop_list.push_back(block_loop<1,2>(output_bispace_indices,
                                            input_bispace_indices,
                                            output_ignore,
                                            input_ignore));

        output_indices_sets.push_back(output_bispace_indices);
        input_indices_sets.push_back(input_bispace_indices);
        output_ignore_sets.push_back(output_ignore);
        input_ignore_sets.push_back(input_ignore);
    }
    block_contract2_kernel<T> bc2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets); 

    sequence<1,T*> output_ptrs((T*)C.get_data_ptr()); 
    sequence<2,const T*> input_ptrs(m_A.get_data_ptr()); 
    input_ptrs[1] = m_B.get_data_ptr();
    sequence<1,sparse_bispace_any_order> output_bispaces(C.get_bispace());
    sequence<2,sparse_bispace_any_order> input_bispaces;
    input_bispaces[0] = m_A.get_bispace();
    input_bispaces[1] = m_B.get_bispace();

    run_loop_list(loop_list,bc2k,output_ptrs,input_ptrs,output_bispaces,input_bispaces);
}

//template<size_t K,size_t M,size_t N,typename T>
//contract_eval_functor<K,M,N,T> contract(letter_expr<K> le,labeled_sparse_btensor<M,T>& A,labeled_sparse_btensor<N,T>& B)
//{
    //return contract_eval_functor<1,M,N,T>(le,A,B);
//}


//Special case for one index contractions
template<size_t M,size_t N,typename T>
contract_eval_functor<1,M,N,T> contract(const letter& a,const labeled_sparse_btensor<M,T>& A,const labeled_sparse_btensor<N,T>& B)
{
    return contract_eval_functor<1,M,N,T>(letter_expr<1>(a),A,B);
}


} // namespace libtensor


#endif /* CONTRACT_H */
