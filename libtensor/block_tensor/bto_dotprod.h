#ifndef LIBTENSOR_BTO_DOTPROD_H
#define LIBTENSOR_BTO_DOTPROD_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_dotprod.h>
#include "bto_traits.h"

namespace libtensor {


/** \brief Computes the dot product of two block tensors
    \tparam N Tensor order.

    The dot product of two tensors is defined as the sum of elements of
    the element-wise product:

    \f[ c = \sum_i a_i b_i \f]

    This operation computes the dot product for a series of arguments.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N, typename T>
class bto_dotprod : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_dotprod< N, bto_traits<T>, bto_dotprod<N, T> > m_gbto;

public:
    /** \brief Initializes the first argument pair
            (identity permutation)
     **/
    bto_dotprod(
            block_tensor_rd_i<N, T> &bt1,
            block_tensor_rd_i<N, T> &bt2) :
        m_gbto(bt1, tensor_transf<N, T>(),
                bt2, tensor_transf<N, T>()) {
    }

    /** \brief Initializes the first argument pair
     **/
    bto_dotprod(
            block_tensor_rd_i<N, T> &bt1, const permutation<N> &perm1,
            block_tensor_rd_i<N, T> &bt2, const permutation<N> &perm2) :
        m_gbto(bt1, tensor_transf<N, T>(perm1),
                bt2, tensor_transf<N, T>(perm2)) {

    }

    /** \brief Adds a pair of arguments (identity permutation)
     **/
    void add_arg(
            block_tensor_rd_i<N, T> &bt1,
            block_tensor_rd_i<N, T> &bt2);

    /** \brief Adds a pair of arguments
     **/
    void add_arg(
            block_tensor_rd_i<N, T> &bt1, const permutation<N> &perm1,
            block_tensor_rd_i<N, T> &bt2, const permutation<N> &perm2);

    /** \brief Returns the dot product of the first argument pair
     **/
    T calculate();

    /** \brief Computes the dot product for all argument pairs
     **/
    void calculate(std::vector<T> &v);
};

template<size_t N>
using btod_dotprod = bto_dotprod<N, double>;

} // namespace libtensor


#endif // LIBTENSOR_BTO_DOTPROD_H
