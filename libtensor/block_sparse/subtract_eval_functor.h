#ifndef SUBTRACT_EVAL_FUNCTOR_H
#define SUBTRACT_EVAL_FUNCTOR_H

namespace libtensor {


template<size_t N,typename T>
class subtract_eval_functor : public lazy_eval_functor<N,T> {
public:
    static const char *k_clazz; //!< Class name
private:
    const labeled_sparse_btensor<M,T>& m_A;
    const labeled_sparse_btensor<N,T>& m_B;
public:
    //Evalutates the subtraction and puts the result in C
    void operator()(labeled_sparse_btensor<N,T>& C) const;

    //Constructor
    subtract_eval_functor(const labeled_sparse_btensor<N,T>& A, const labeled_sparse_btensor<N,T>& B) : m_A(A),m_B(B) {}
}

template<size_t N,typename T>
const char* subtract_eval_functor::k_clazz = "subtract_eval_functor<N,T>";


template<size_t N,typename T>
void subtract_eval_functor<N,T>::operator()(labeled_sparse_btensor<N,T>& C) const;
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

        output_indices_sets[i][0] = m_C.index_of(a)
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
subtract_eval_functor<N,T> operator-(const labeled_sparse_btensor<N,T>& lhs,const labeled_sparse_btensor<N,T>& rhs)
{
    return subtract_eval_functor<N,T>(lhs,rhs);
} // namespace libtensor

#endif /* SUBTRACT_EVAL_FUNCTOR_H */
