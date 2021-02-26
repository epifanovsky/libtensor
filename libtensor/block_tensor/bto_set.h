#ifndef LIBTENSOR_BTO_SET_H
#define LIBTENSOR_BTO_SET_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_set.h>
#include "bto_traits.h"

namespace libtensor {


/** \brief Sets all elements of a block tensor to a value preserving
        the symmetry
    \tparam N Tensor order.

    \sa gen_bto_set

    \ingroup libtensor_block_tensor_bto
 **/
template<size_t N, typename T>
class bto_set : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    gen_bto_set< N, bto_traits<T>, bto_set<N, T> > m_gbto;

public:
    /** \brief Initializes the operation
        \param v Value to be assigned to the tensor elements.
     **/
    bto_set(T v = 0.0) :
        m_gbto(v)
    { }

    /** \brief Performs the operation
        \param bta Output block tensor.
     **/
    void perform(block_tensor_wr_i<N, T> &bta) {

        m_gbto.perform(bta);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_SET_H
