#ifndef LIBTENSOR_CTF_BTOD_RANDOM_H
#define LIBTENSOR_CTF_BTOD_RANDOM_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_bto_random.h>
#include "ctf_btod_traits.h"

namespace libtensor {


/** \brief Fills a distributed block tensor with random data without affecting
        its symmetry
    \tparam N Block tensor order.

    \sa gen_bto_random, btod_random

    \ingroup libtensor_ctf_block_tensor
 **/
template<size_t N>
class ctf_btod_random : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    gen_bto_random< N, ctf_btod_traits, ctf_btod_random<N> > m_gbto;

public:
    /** \brief Fills a block tensor with random values preserving
            symmetry
        \param bt Block %tensor.
     **/
    void perform(ctf_block_tensor_i<N, double> &bt);

    /** \brief Fills one block of a block tensor with random values
            preserving symmetry
        \param bt Block tensor.
        \param idx Block index in the block tensor.
     **/
    void perform(ctf_block_tensor_i<N, double> &bt, const index<N> &idx);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_BTOD_RANDOM_H
