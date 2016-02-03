#ifndef LIBTENSOR_CTF_BTOD_SET_H
#define LIBTENSOR_CTF_BTOD_SET_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_set.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Sets all elements of a block tensor to a value preserving
        the symmetry
    \tparam N Tensor order.

    \sa gen_bto_set, btod_set

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_set : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    gen_bto_set< N, ctf_btod_traits, ctf_btod_set<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    ctf_btod_set(double v = 0.0) :
        m_gbto(v)
    { }

    /** \brief Performs the operation
        \param bta Output block tensor.
     **/
    void perform(ctf_block_tensor_i<N, double> &bta);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_SET_H
