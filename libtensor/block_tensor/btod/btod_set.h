#ifndef LIBTENSOR_BTOD_SET_H
#define LIBTENSOR_BTOD_SET_H

#include <libtensor/block_tensor/bto/bto_set.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {


/** \brief Sets all elements of a block tensor to a value preserving
        the symmetry
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_set : public bto_set<N, btod_traits> {
public:
    static const char *k_clazz; //!< Class name

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    btod_set(double v = 0.0) : bto_set<N, btod_traits>(v) { }

    /** \brief Virtual destructor
     **/
    virtual ~btod_set() { }

};


template<size_t N>
const char *btod_set<N>::k_clazz = "btod_set<N>";


} // namespace libtensor

#endif // LIBTENSOR_BTOD_SET_H
