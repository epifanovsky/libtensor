#ifndef LIBTENSOR_BTOD_DOTPROD_H
#define LIBTENSOR_BTOD_DOTPROD_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_dotprod.h>
#include "btod_traits.h"

namespace libtensor {


/** \brief Computes the dot product of two block tensors
    \tparam N Tensor order.

    The dot product of two tensors is defined as the sum of elements of
    the element-wise product:

    \f[ c = \sum_i a_i b_i \f]

    This operation computes the dot product for a series of arguments.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N>
class btod_dotprod : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_dotprod< N, btod_traits, btod_dotprod<N> > m_gbto;

public:
    /** \brief Initializes the first argument pair
            (identity permutation)
     **/
    btod_dotprod(
            block_tensor_rd_i<N, double> &bt1,
            block_tensor_rd_i<N, double> &bt2) :
        m_gbto(bt1, tensor_transf<N, double>(),
                bt2, tensor_transf<N, double>()) {
    }

    /** \brief Initializes the first argument pair
     **/
    btod_dotprod(
            block_tensor_rd_i<N, double> &bt1, const permutation<N> &perm1,
            block_tensor_rd_i<N, double> &bt2, const permutation<N> &perm2) :
        m_gbto(bt1, tensor_transf<N, double>(perm1),
                bt2, tensor_transf<N, double>(perm2)) {

    }

    /** \brief Adds a pair of arguments (identity permutation)
     **/
    void add_arg(
            block_tensor_rd_i<N, double> &bt1,
            block_tensor_rd_i<N, double> &bt2);

    /** \brief Adds a pair of arguments
     **/
    void add_arg(
            block_tensor_rd_i<N, double> &bt1, const permutation<N> &perm1,
            block_tensor_rd_i<N, double> &bt2, const permutation<N> &perm2);

    /** \brief Returns the dot product of the first argument pair
     **/
    double calculate();

    /** \brief Computes the dot product for all argument pairs
     **/
    void calculate(std::vector<double> &v);
};


} // namespace libtensor


#endif // LIBTENSOR_BTOD_DOTPROD_H
