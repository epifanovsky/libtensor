#ifndef LIBTENSOR_TOD_ADD_H
#define LIBTENSOR_TOD_ADD_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>

namespace libtensor {


/** \brief Adds a series of tensors

    Addition of n tensors:
    \f[
        B = \left( c_1 \mathcal{P}_1 A_1 + c_2 \mathcal{P}_2 A_2 + \cdots +
            c_n \mathcal{P}_n A_n \right)
    \f]

    Each operand must have the same dimensions as the result in order
    for the operation to be successful.

    \ingroup libtensor_dense_tensor_tod
 **/
template<size_t N>
class tod_add : public timings< tod_add<N> > {
public:
    static const char* k_clazz; //!< Class name

private:
    struct arg {
        dense_tensor_rd_i<N, double> &t;
        permutation<N> p;
        double c;
        arg(dense_tensor_rd_i<N, double> &t_, const permutation<N> &p_,
            double c_ = 1.0) : t(t_), p(p_), c(c_) { }
    };

private:
    std::list<arg> m_args; //!< List of all tensors to add
    dimensions<N> m_dims;  //!< Dimensions of the output tensor

public:
    /** \brief Initializes the addition operation
        \param t First tensor in the series.
        \param c Scaling coefficient.
     **/
    tod_add(dense_tensor_rd_i<N, double> &t, double c = 1.0);

    /** \brief Initializes the addition operation
        \param t First tensor in the series.
        \param perm Permutation of the first tensor.
        \param c Scaling coefficient.
     **/
    tod_add(dense_tensor_rd_i<N, double> &t, const permutation<N> &perm,
        double c = 1.0);

    /** \brief Adds an operand
        \param t Tensor.
        \param c Coefficient.
     **/
    void add_op(dense_tensor_rd_i<N, double> &t, double c);

    /** \brief Adds an operand
        \param t Tensor.
        \param perm Permutation.
        \param c Coefficient.
     **/
    void add_op(dense_tensor_rd_i<N, double> &t, const permutation<N> &perm,
        double c);

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    /** \brief Performs the operation
     **/
    void perform(bool zero, double c, dense_tensor_wr_i<N, double> &tb);

private:
    /** \brief Adds an operand (internal)
     **/
    void add_operand(dense_tensor_rd_i<N, double> &t,
        const permutation<N> &perm, double c);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_ADD_H
