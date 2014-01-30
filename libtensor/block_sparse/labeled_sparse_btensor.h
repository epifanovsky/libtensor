#ifndef LABELED_SPARSE_BTENSOR_H
#define LABELED_SPARSE_BTENSOR_H

#include "sparse_btensor.h"
#include "gen_labeled_btensor.h"
#include "block_permute_kernel.h"
#include "batch_provider_factory.h"

namespace libtensor {

//Forward declaration for member variable
template<size_t N,typename T>
class sparse_btensor;

template<size_t N,typename T = double> 
class labeled_sparse_btensor : public gen_labeled_btensor<N,T>
{
private:
    sparse_btensor<N,T>& m_tensor; 
    letter_expr<N> m_le;
public:

    labeled_sparse_btensor(sparse_btensor<N,T>& tensor,const letter_expr<N>& le) : m_tensor(tensor),m_le(le) { };

    letter_expr<N> get_letter_expr() const { return m_le; }
    const T* get_data_ptr() const { return m_tensor.get_data_ptr(); }

    /** \brief Return the sparse_bispace defining this tensor 
     **/
    sparse_bispace<N> get_bispace() const { return m_tensor.get_bispace(); }; 

    //TODO - could template this later to allow assignment across tensor formats
    //For permutations
    labeled_sparse_btensor<N,T>& operator=(const labeled_sparse_btensor<N,T>& rhs);

    //Store the result of an expression in this tensor
    labeled_sparse_btensor<N,T>& operator=(const batch_provider_factory<N,T>& factory);
};

template<size_t N,typename T> 
labeled_sparse_btensor<N,T>& labeled_sparse_btensor<N,T>::operator=(const labeled_sparse_btensor<N,T>& rhs)
{
    //Determine the permutation of indices between the two tensors
    //We also populate the loops necessary to execute the transformation
	std::vector<size_t> permutation_entries(N);
    std::vector< sparse_bispace_any_order > bispaces(1,this->m_tensor.get_bispace());
    bispaces.push_back(rhs.m_tensor.get_bispace());

    std::vector<block_loop> loops;
    for(size_t i = 0; i < N; ++i)
    {
        const letter& a = m_le.letter_at(i);
        size_t rhs_idx = rhs.m_le.index_of(a);
		permutation_entries[i] = rhs_idx;

        //Populate the loop for this index
		block_loop bl(bispaces);
		bl.set_subspace_looped(0,i);
		bl.set_subspace_looped(1,rhs_idx);
		loops.push_back(bl);
    }
    sparse_loop_list sll(loops);
    runtime_permutation perm(permutation_entries);
    block_permute_kernel<T> bpk(perm);

    //Deliberately case away the const
    std::vector<T*> ptrs(1,(T*)this->m_tensor.get_data_ptr());
    ptrs.push_back((T*)rhs.m_tensor.get_data_ptr());
    sll.run(bpk,ptrs);
    return *this;
}

//Used for evaluating contractions, to prevent an unnecessary copy at the end
template<size_t N,typename T>
labeled_sparse_btensor<N,T>& labeled_sparse_btensor<N,T>::operator=(const batch_provider_factory<N,T>& factory)
{
    batch_provider<T>* bp = factory.get_batch_provider(*this);
    bp->get_batch((T*)m_tensor.get_data_ptr());
    delete bp;
    return *this;
}


#if 0
//The binary operator we need
template<size_t N,typename T>
subtract_eval_functor<N,T> operator-(const labeled_sparse_btensor<N,T>& lhs,const labeled_sparse_btensor<N,T>& rhs)
{
    return subtract_eval_functor<N,T>(lhs,rhs);
}
#endif

} // namespace libtensor

#endif /* LABELED_SPARSE_BTENSOR_H */
