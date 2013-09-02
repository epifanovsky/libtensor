#ifndef LIBTENSOR_BTOD_SET_H
#define LIBTENSOR_BTOD_SET_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_set.h>
#include "btod_traits.h"

namespace libtensor {


/** \brief Sets all elements of a block tensor to a value preserving
        the symmetry
    \tparam N Tensor order.

    \sa gen_bto_set

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N>
class btod_set : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_set< N, btod_traits, btod_set<N> > m_gbto;

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    btod_set(double v = 0.0) :
        m_gbto(v)
    { }

    /** \brief Performs the operation
        \param bta Output block tensor.
     **/
    void perform(block_tensor_i<N, double> &bta) {

        m_gbto.perform(bta);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_H
