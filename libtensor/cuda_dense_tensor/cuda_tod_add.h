#ifndef LIBTENSOR_CUDA_TOD_ADD_H
#define LIBTENSOR_CUDA_TOD_ADD_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/cuda_dense_tensor/cuda_dense_tensor_i.h>

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
class cuda_tod_add : public timings< cuda_tod_add<N> >, public noncopyable {
public:
    static const char* k_clazz; //!< Class name

private:
    struct arg {
        cuda_dense_tensor_i<N, double> &t;
        permutation<N> p;
        double c;
        arg(cuda_dense_tensor_i<N, double> &t_, const permutation<N> &p_,
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
    cuda_tod_add(cuda_dense_tensor_i<N, double> &t, double c = 1.0);

    /** \brief Initializes the addition operation
        \param t First %tensor in the series.
        \param p Permutation of the first %tensor.
        \param c Scaling coefficient.
     **/
    cuda_tod_add(cuda_dense_tensor_i<N, double> &t, const permutation<N> &p,
        double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~cuda_tod_add();

    //@}


    //! \name Operation
    //@{

    /** \brief Adds an operand
        \param t Tensor.
        \param c Coefficient.
     **/
    void add_op(cuda_dense_tensor_i<N, double> &t, double c);

    /** \brief Adds an operand
        \param t Tensor.
        \param p Permutation of %tensor elements.
        \param c Coefficient.
     **/
    void add_op(cuda_dense_tensor_i<N, double> &t, const permutation<N> &p, double c);

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    /** \brief Computes the sum into the output tensor
     **/
    void perform(bool zero, double c, cuda_dense_tensor_i<N, double> &tb);

    //@}

private:
    /** \brief Adds an operand (implementation)
     **/
    void add_operand(cuda_dense_tensor_i<N, double> &t, const permutation<N> &perm,
        double c);

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_TOD_ADD_H
