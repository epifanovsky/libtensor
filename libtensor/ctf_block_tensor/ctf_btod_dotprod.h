#ifndef LIBTENSOR_CTF_BTOD_DOTPROD_H
#define LIBTENSOR_CTF_BTOD_DOTPROD_H

#include <vector>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_dotprod.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Computes the dot product of two distributed block tensors
    \tparam N Tensor order.

    \sa gen_bto_dotprod, btod_dotprod

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_dotprod : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef typename ctf_btod_traits::bti_traits bti_traits;

private:
    gen_bto_dotprod< N, ctf_btod_traits, ctf_btod_dotprod<N> > m_gbto;

public:
    /** \brief Initializes the first argument pair
            (identity permutation)
     **/
    ctf_btod_dotprod(
        ctf_block_tensor_rd_i<N, double> &bt1,
        ctf_block_tensor_rd_i<N, double> &bt2) :

        m_gbto(bt1, tensor_transf<N, double>(), bt2,
            tensor_transf<N, double>()) {

    }

    /** \brief Initializes the first argument pair
     **/
    ctf_btod_dotprod(
        ctf_block_tensor_rd_i<N, double> &bt1,
        const permutation<N> &perm1,
        ctf_block_tensor_rd_i<N, double> &bt2,
        const permutation<N> &perm2) :

        m_gbto(bt1, tensor_transf<N, double>(perm1), bt2,
            tensor_transf<N, double>(perm2)) {

    }

    /** \brief Virtual destructor
     **/
    virtual ~ctf_btod_dotprod() { }

    /** \brief Adds a pair of arguments (identity permutation)
     **/
    void add_arg(
        ctf_block_tensor_rd_i<N, double> &bt1,
        ctf_block_tensor_rd_i<N, double> &bt2);

    /** \brief Adds a pair of arguments
     **/
    void add_arg(
        ctf_block_tensor_rd_i<N, double> &bt1,
        const permutation<N> &perm1,
        ctf_block_tensor_rd_i<N, double> &bt2,
        const permutation<N> &perm2);

    /** \brief Returns the dot product of the first argument pair
     **/
    double calculate();

    /** \brief Computes the dot product for all argument pairs
     **/
    void calculate(std::vector<double> &v);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_DOTPROD_H
