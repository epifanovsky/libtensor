#ifndef LABELED_SPARSE_BTENSOR_H
#define LABELED_SPARSE_BTENSOR_H

#include "sparse_btensor.h"


namespace libtensor
{

//Forward declaration for member variable
template<size_t N,typename T>
class sparse_btensor;

//TODO: Make templated labeledXXX classs to unify tensor types
//For labeling general tensors
template<size_t N,typename T = double> 
class labeled_sparse_btensor
{
private:
    sparse_btensor<N,T>& m_tensor; 
    letter_expr<N> m_le;
public:

    labeled_sparse_btensor(sparse_btensor<N,T>& tensor,const letter_expr<N>& le) : m_tensor(tensor),m_le(le) { };

    //TODO - could template this later to allow assignment across tensor formats
    labeled_sparse_btensor<N,T>& operator=(const labeled_sparse_btensor<N,T>& rhs);
};


template<size_t N,typename T> 
labeled_sparse_btensor<N,T>& labeled_sparse_btensor<N,T>::operator=(const labeled_sparse_btensor<N,T>& rhs)
{
    //Determine the permutation of indices between the two tensors
    //Permutation is defined as acting on the RHS tensor to produce the THIS tensor
    //We also populate the loops necessary to execute the transformation
    permute_map pm;
    std::vector< block_loop<1,1> >  loop_list;
    for(size_t i = 0; i < N; ++i)
    {
        const letter& a = m_le.letter_at(i);
        size_t rhs_idx = rhs.m_le.index_of(a);
        if(rhs_idx != i)
        {
            pm.insert(std::make_pair(rhs_idx,i));
        }

        //Populate the loop for this index
        loop_list.push_back(block_loop<1,1>(sequence<1,size_t>(i),sequence<1,size_t>(rhs_idx),sequence<1,bool>(false),sequence<1,bool>(false)));
    }

    block_permute_kernel<T> bpk(pm);

    //Deliberately case away the const
    sequence<1,T*> output_ptrs((T*)this->m_tensor.get_data_ptr());
    //TODO: should use const on input ptrs always
    sequence<1,T*> input_ptrs((T*)rhs.m_tensor.get_data_ptr());

    const sparse_bispace<N>& spb_1 = this->m_tensor.get_bispace();
    const sparse_bispace<N>& spb_2 = rhs.m_tensor.get_bispace();

    sequence<1, sparse_bispace_any_order> output_bispaces(spb_1);
    sequence<1, sparse_bispace_any_order> input_bispaces(spb_2);
    run_loop_list(loop_list,bpk,output_ptrs,input_ptrs,output_bispaces,input_bispaces);
}

} // namespace libtensor

#endif /* LABELED_SPARSE_BTENSOR_H */
