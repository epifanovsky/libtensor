#ifndef LIBTENSOR_DIAG_BTOD_DOTPROD_H
#define LIBTENSOR_DIAG_BTOD_DOTPROD_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_dotprod.h>
#include <libtensor/diag_block_tensor/diag_block_tensor_i.h>
#include "diag_btod_traits.h"

namespace libtensor {


/** \brief Computes the dot product of two diagonal block tensors
    \tparam N Tensor order.

    \sa btod_dotprod

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N>
class diag_btod_dotprod : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    gen_bto_dotprod< N, diag_btod_traits, diag_btod_dotprod<N> > m_gbto;

public:
    /** \brief Initializes the first argument pair
            (identity permutation)
     **/
    diag_btod_dotprod(
        diag_block_tensor_rd_i<N, double> &bt1,
        diag_block_tensor_rd_i<N, double> &bt2) :

        m_gbto(bt1, tensor_transf<N, double>(),
            bt2, tensor_transf<N, double>()) {
    }

    /** \brief Initializes the first argument pair
     **/
    diag_btod_dotprod(
        diag_block_tensor_rd_i<N, double> &bt1, const permutation<N> &perm1,
        diag_block_tensor_rd_i<N, double> &bt2, const permutation<N> &perm2) :

        m_gbto(bt1, tensor_transf<N, double>(perm1),
            bt2, tensor_transf<N, double>(perm2)) {

    }

    /** \brief Adds a pair of arguments (identity permutation)
     **/
    void add_arg(
        diag_block_tensor_rd_i<N, double> &bt1,
        diag_block_tensor_rd_i<N, double> &bt2);

    /** \brief Adds a pair of arguments
     **/
    void add_arg(
        diag_block_tensor_rd_i<N, double> &bt1, const permutation<N> &perm1,
        diag_block_tensor_rd_i<N, double> &bt2, const permutation<N> &perm2);

    /** \brief Returns the dot product of the first argument pair
     **/
    double calculate();

    /** \brief Computes the dot product for all argument pairs
     **/
    void calculate(std::vector<double> &v);

};


} // namespace libtensor


#endif // LIBTENSOR_DIAG_BTOD_DOTPROD_H
