#ifndef LIBTENSOR_BTOD_RANDOM_H
#define LIBTENSOR_BTOD_RANDOM_H

#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/gen_block_tensor/gen_bto_random.h>
#include "block_tensor_i.h"

namespace libtensor {


/** \brief Fills a block %tensor with random data without affecting its
        %symmetry
    \tparam N Block %tensor order.

    \ingroup libtensor_block_tensor_btod
 **/
template<size_t N>
class btod_random : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    gen_bto_random< N, btod_traits, btod_random<N> > m_gbto;

public:
    /** \brief Fills a block %tensor with random values preserving
            symmetry
        \param bt Block %tensor.
     **/
    void perform(block_tensor_wr_i<N, double> &bt);

    /** \brief Fills one block of a block %tensor with random values
            preserving symmetry
        \param bt Block %tensor.
        \param idx Block %index in the block %tensor.
     **/
    void perform(block_tensor_wr_i<N, double> &bt, const index<N> &idx);
};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_RANDOM_H
