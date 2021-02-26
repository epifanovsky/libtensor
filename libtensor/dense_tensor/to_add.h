#ifndef LIBTENSOR_TO_ADD_H
#define LIBTENSOR_TO_ADD_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>

namespace libtensor {


/** \brief Adds a series of tensors

    Addition of n tensors:
    \f[
        B = \left( \hat{T}_1 A_1 + \hat{T}_2 A_2 + \cdots +
            \hat{T}_n A_n \right)
    \f]
    where the \f$ \hat{T}_i \f$ are tensor transformations consisting of
    a permutation \f$ \mathcal{P}_i \f$ and a scalar transformation
    \f$ \mathcal{S}_i \f$ . Each operand must have the same dimensions as
    the result in order for the operation to be successful.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N, typename T>
class to_add : public timings< to_add<N, T> >, public noncopyable {
public:
    static const char* k_clazz; //!< Class name

    typedef tensor_transf<N, T> tensor_transf_t;

private:
    struct arg {
        dense_tensor_rd_i<N, T> &t;
        permutation<N> perm;
        T c;
        arg(dense_tensor_rd_i<N, T> &t_,
                const permutation<N> &perm_, T c_) :
            t(t_), perm(perm_), c(c_) { }
    };

private:
    std::list<arg> m_args; //!< List of all tensors to add
    dimensions<N> m_dims;  //!< Dimensions of the output tensor

public:
    /** \brief Initializes the addition operation
        \param t First tensor in the series.
        \param tr Tensor transformation
     **/
    to_add(dense_tensor_rd_i<N, T> &t,
            const tensor_transf_t &tr = tensor_transf_t());

    /** \brief Initializes the addition operation
        \param t First tensor in the series.
        \param c Scaling coefficient.
     **/
    to_add(dense_tensor_rd_i<N, T> &t, T c);

    /** \brief Initializes the addition operation
        \param t First tensor in the series.
        \param perm Permutation of the first tensor.
        \param c Scaling coefficient.
     **/
    to_add(dense_tensor_rd_i<N, T> &t, const permutation<N> &perm,
        T c = 1.0);

    /** \brief Adds an operand
        \param t Tensor.
        \param tr Tensor transformation.
     **/
    void add_op(dense_tensor_rd_i<N, T> &t,
            const tensor_transf_t &tr = tensor_transf_t());

    /** \brief Adds an operand
        \param t Tensor.
        \param c Coefficient.
     **/
    void add_op(dense_tensor_rd_i<N, T> &t, T c);

    /** \brief Adds an operand
        \param t Tensor.
        \param perm Permutation.
        \param c Coefficient.
     **/
    void add_op(dense_tensor_rd_i<N, T> &t,
            const permutation<N> &perm, T c = 1.0);

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    /** \brief Performs the operation
        \param zero Zero result first
        \param tb Tensor to add result to
     **/
    void perform(bool zero, dense_tensor_wr_i<N, T> &tb);

private:
    /** \brief Adds an operand (internal)
     **/
    void add_operand(dense_tensor_rd_i<N, T> &t,
        const permutation<N> &perm, T c);

};


} // namespace libtensor

#endif // LIBTENSOR_TO_ADD_H
