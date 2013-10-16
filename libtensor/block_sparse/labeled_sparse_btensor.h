#ifndef LABELED_SPARSE_BTENSOR_H
#define LABELED_SPARSE_BTENSOR_H

#include "sparse_btensor.h"
#include "lazy_eval_functor.h"

//TODO: REMOVE
#include "contract.h"


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

    /** \brief Returns the %letter at a given position
        \throw out_of_bounds If the %index is out of bounds.
     **/
    const letter &letter_at(size_t i) const throw(out_of_bounds) {
        return m_le.letter_at(i);
    }

    /** \brief Returns whether the associated expression contains a %letter
     **/
    bool contains(const letter &let) const {
        return m_le.contains(let);
    }

    /** \brief Returns the %index of a %letter in the expression
        \throw exception If the expression doesn't contain the %letter.
     **/
    size_t index_of(const letter &let) const throw(exception) {
        return m_le.index_of(let);
    }

    const T* get_data_ptr() const { return m_tensor.get_data_ptr(); }

    /** \brief Return the sparse_bispace defining this tensor 
     **/
    const sparse_bispace<N>& get_bispace() const { return m_tensor.get_bispace(); }; 

    //TODO - could template this later to allow assignment across tensor formats
    //For permutations
    labeled_sparse_btensor<N,T>& operator=(const labeled_sparse_btensor<N,T>& rhs);

    //For contractions
    labeled_sparse_btensor<N,T>& operator=(const lazy_eval_functor<N,T>& functor);
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
    sequence<1,const T*> input_ptrs(rhs.m_tensor.get_data_ptr());

    const sparse_bispace<N>& spb_1 = this->m_tensor.get_bispace();
    const sparse_bispace<N>& spb_2 = rhs.m_tensor.get_bispace();

    sequence<1, sparse_bispace_any_order> output_bispaces(spb_1);
    sequence<1, sparse_bispace_any_order> input_bispaces(spb_2);
    run_loop_list(loop_list,bpk,output_ptrs,input_ptrs,output_bispaces,input_bispaces);
    return *this;
}

//Used for evaluating contractions, to prevent an unnecessary copy at the end
template<size_t N,typename T>
labeled_sparse_btensor<N,T>& labeled_sparse_btensor<N,T>::operator=(const lazy_eval_functor<N,T>& functor)
{
    functor(*this);
    return *this; 
}

} // namespace libtensor

#endif /* LABELED_SPARSE_BTENSOR_H */
