#ifndef LIBTENSOR_CTF_BTOD_SCALE_H
#define LIBTENSOR_CTF_BTOD_SCALE_H

#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/gen_block_tensor/gen_bto_scale.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Scales a CTF block tensor by a coefficient
    \tparam N Tensor order.

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_scale : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    gen_bto_scale< N, ctf_btod_traits, ctf_btod_scale<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param bt Block tensor.
        \param c Scaling coefficient.
     **/
    ctf_btod_scale(
        ctf_block_tensor_i<N, double> &bt,
        const scalar_transf<double> &c) :

        m_gbto(bt, c)
    { }

    /** \brief Initializes the operation
        \param bt Block tensor.
        \param c Scaling coefficient.
     **/
    ctf_btod_scale(
        ctf_block_tensor_i<N, double> &bt,
        double c) :

        m_gbto(bt, scalar_transf<double>(c))
    { }

    /** \brief Performs the operation
     **/
    void perform() {

        m_gbto.perform();
    }

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SCALE_H
