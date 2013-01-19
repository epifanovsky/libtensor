#ifndef LIBTENSOR_CUDA_TOD_COPY_H
#define LIBTENSOR_CUDA_TOD_COPY_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/dense_tensor/dense_tensor_i.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/core/bad_dimensions.h>

namespace libtensor {


/** \brief Makes a copy of a tensor, scales or permutes tensor elements
        if necessary
    \tparam N Tensor order.

    This operation makes a scaled and permuted copy of a tensor.
    The result can replace or be added to the output tensor.

    \sa tod_copy

    \ingroup libtensor_cuda_tod
 **/
template<size_t N>
class cuda_tod_copy : public timings< cuda_tod_copy<N> >, public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    dense_tensor_i<N, double> &m_ta; //!< Source %tensor
    permutation<N> m_perm; //!< Permutation of elements
    double m_c; //!< Scaling coefficient
    dimensions<N> m_dimsb; //!< Dimensions of output %tensor

public:
    /** \brief Prepares the copy operation
        \param ta Source tensor.
        \param c Coefficient.
     **/
    cuda_tod_copy(dense_tensor_i<N, double> &ta, double c = 1.0);

    /** \brief Prepares the permute & copy operation
        \param ta Source tensor.
        \param p Permutation of tensor elements.
        \param c Coefficient.
     **/
    cuda_tod_copy(dense_tensor_i<N, double> &ta, const permutation<N> &p,
        double c = 1.0);

    /** \brief Virtual destructor
     **/
    virtual ~cuda_tod_copy() { }

    virtual void prefetch();

    virtual void perform(dense_tensor_i<N, double> &t);

    virtual void perform(dense_tensor_i<N, double> &t, double c);

private:
    static dimensions<N> mk_dimsb(dense_tensor_i<N, double> &ta,
        const permutation<N> &perm);

    void do_perform(dense_tensor_i<N, double> &t, double c);

};


} // namespace libtensor

#endif // LIBTENSOR_CUDA_TOD_COPY_H
