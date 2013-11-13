#ifndef LABELED_SPARSE_BTENSOR_H
#define LABELED_SPARSE_BTENSOR_H

#include "sparse_btensor.h"
#include "lazy_eval_functor.h"

namespace libtensor {

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
    //We also populate the loops necessary to execute the transformation
	std::vector<size_t> permutation_entries(N);
    std::vector< block_loop<1,1> >  loop_list;
    for(size_t i = 0; i < N; ++i)
    {
        const letter& a = m_le.letter_at(i);
        size_t rhs_idx = rhs.m_le.index_of(a);
		permutation_entries[i] = rhs_idx;

        //Populate the loop for this index
        loop_list.push_back(block_loop<1,1>(sequence<1,size_t>(i),sequence<1,size_t>(rhs_idx),sequence<1,bool>(false),sequence<1,bool>(false)));
    }
    runtime_permutation perm(permutation_entries);
    block_permute_kernel<T> bpk(perm);

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

//TODO put this in its own file with forward declaration shizzz
template<size_t N,typename T>
class subtract_eval_functor : public lazy_eval_functor<N,T> {
public:
    static const char *k_clazz; //!< Class name
private:
    const labeled_sparse_btensor<N,T>& m_A;
    const labeled_sparse_btensor<N,T>& m_B;
public:
    //Evalutates the subtraction and puts the result in C
    void operator()(labeled_sparse_btensor<N,T>& C) const;

    //For direct tensors
    virtual void operator()(labeled_sparse_btensor<N,T>& dest,std::vector< block_list >& output_block_lists) const {
        throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                __FILE__, __LINE__, "not implemented");
    }

    //Constructor
    subtract_eval_functor(const labeled_sparse_btensor<N,T>& A, const labeled_sparse_btensor<N,T>& B) : m_A(A),m_B(B) {}
};

template<size_t N,typename T>
const char* subtract_eval_functor<N,T>::k_clazz = "subtract_eval_functor<N,T>";


template<size_t N,typename T>
void subtract_eval_functor<N,T>::operator()(labeled_sparse_btensor<N,T>& C) const
{
    //Build the loops for the subtraction 
    std::vector< block_loop<1,2> > loop_list;
    std::vector< sequence<1,size_t> > output_indices_sets(N);
    std::vector< sequence<2,size_t> > input_indices_sets(N);

    //Nothing can be ignored in a subtraction expressino
    std::vector< sequence<1,bool> > output_ignore_sets(N,sequence<1,bool>(false));
    std::vector< sequence<2,bool> > input_ignore_sets(N,sequence<2,bool>(false));

    for(size_t i = 0; i < N; ++i)
    {
        const letter& a = C.letter_at(i);
        if((!m_A.contains(a)) || (!m_B.contains(a)))
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "Indices are not consistent");

        }

        //TODO: REMOVE THIS REQUIREMENT TIME CRUNCH WANT AN EASY SUBTRACTION KERNEL
        if((m_A.index_of(a) != i) || (m_B.index_of(a) != i))
        {
            throw bad_parameter(g_ns, k_clazz,"operator()(...)",
                    __FILE__, __LINE__, "Indices must be in order");
        }

        output_indices_sets[i][0] = C.index_of(a);
        input_indices_sets[i][0] = m_A.index_of(a);
        input_indices_sets[i][1] = m_B.index_of(a);

        loop_list.push_back(block_loop<1,2>(output_indices_sets[i],
                                            input_indices_sets[i],
                                            output_ignore_sets[i],
                                            input_ignore_sets[i]));
    }
    block_subtract2_kernel<T> bs2k(output_indices_sets,input_indices_sets,output_ignore_sets,input_ignore_sets);

    sequence<1,T*> output_ptrs((T*)C.get_data_ptr()); 
    sequence<2,const T*> input_ptrs(m_A.get_data_ptr()); 
    input_ptrs[1] = m_B.get_data_ptr();
    sequence<1,sparse_bispace_any_order> output_bispaces(C.get_bispace());
    sequence<2,sparse_bispace_any_order> input_bispaces;
    input_bispaces[0] = m_A.get_bispace();
    input_bispaces[1] = m_B.get_bispace();

    run_loop_list(loop_list,bs2k,output_ptrs,input_ptrs,output_bispaces,input_bispaces);

}

//The binary operator we need
template<size_t N,typename T>
subtract_eval_functor<N,T> operator-(const labeled_sparse_btensor<N,T>& lhs,const labeled_sparse_btensor<N,T>& rhs)
{
    return subtract_eval_functor<N,T>(lhs,rhs);
} // namespace libtensor

} // namespace libtensor

#endif /* LABELED_SPARSE_BTENSOR_H */
