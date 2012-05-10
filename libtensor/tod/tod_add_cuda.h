#ifndef LIBTENSOR_TOD_ADD_CUDA_H
#define LIBTENSOR_TOD_ADD_CUDA_H

#include <list>
#include "tod_cuda_copy.h"
#include "bad_dimensions.h"

namespace libtensor {


/** \brief Adds a series of tensors

    Tensor addition of n tensors:
    \f[
        B = \left( c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots +
        c_n \mathcal{P}_n A_n \right) \f]

    Each operand must have the same dimensions as the result in order
    for the operation to be successful.

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_add_cuda : public timings< tod_add_cuda<N> > {
public:
    static const char* k_clazz; //!< Class name

private:
    struct arg {
        dense_tensor_i<N, double> &t;
        permutation<N> p;
        double c;
        arg(dense_tensor_i<N, double> &t_, const permutation<N> &p_,
            double c_ = 1.0) : t(t_), p(p_), c(c_) { }
    };

private:
    std::list<arg> m_args; //!< List of all operands to add
    dimensions<N> m_dims;  //!< Dimensions of the output tensor

public:
    //! \name Construction and destruction
    //@{

    /** \brief Initializes the addition operation
        \param t First %tensor in the series.
        \param c Scaling coefficient.
     **/
    tod_add_cuda(dense_tensor_i<N, double> &t, double c = 1.0);

    /** \brief Initializes the addition operation
        \param t First %tensor in the series.
        \param p Permutation of the first %tensor.
        \param c Scaling coefficient.
     **/
    tod_add_cuda(dense_tensor_i<N, double> &t, const permutation<N> &p,
        double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~tod_add_cuda();

    //@}


    //! \name Operation
    //@{

    /** \brief Adds an operand
        \param t Tensor.
        \param c Coefficient.
     **/
    void add_op(dense_tensor_i<N, double> &t, double c);

    /** \brief Adds an operand
        \param t Tensor.
        \param p Permutation of %tensor elements.
        \param c Coefficient.
     **/
    void add_op(dense_tensor_i<N, double> &t, const permutation<N> &p, double c);

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    void perform(bool zero, double c, dense_tensor_i<N, double> &tb);

    /** \brief Computes the sum into the output %tensor
     **/
//    void perform(dense_tensor_i<N, double> &t);

    /** \brief Adds the sum to the output %tensor
     **/
//    void perform(dense_tensor_i<N, double> &t, double c);

    //@}

private:
    /** \brief Adds an operand (implementation)
     **/
    void add_operand(dense_tensor_i<N, double> &t, const permutation<N> &perm,
        double c);

};


} // namespace libtensor


#ifdef LIBTENSOR_INSTANTIATE_TEMPLATES

namespace libtensor {

    extern template class tod_add_cuda<1>;
    extern template class tod_add_cuda<2>;
    extern template class tod_add_cuda<3>;
    extern template class tod_add_cuda<4>;
    extern template class tod_add_cuda<5>;
    extern template class tod_add_cuda<6>;

} // namespace libtensor

#else // LIBTENSOR_INSTANTIATE_TEMPLATES
#include "tod_add_cuda_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES


#endif // LIBTENSOR_TOD_ADD_CUDA_H
