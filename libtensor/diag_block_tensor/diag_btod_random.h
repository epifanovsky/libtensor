#ifndef LIBTENSOR_DIAG_BTOD_RANDOM_H
#define LIBTENSOR_DIAG_BTOD_RANDOM_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include "diag_block_tensor_i.h"

namespace libtensor {


/** \brief Fills a diagonal block tensor with random data without affecting its
        symmetry
    \tparam T Block tensor order.

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N>
class diag_btod_random :
    public timings< diag_btod_random<N> >,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    /** \brief Virtual destructor
     **/
    virtual ~diag_btod_random() { }

    /** \brief Fills a diagonal block tensor with random values preserving
            symmetry
        \param bt Block tensor.
     **/
    void perform(diag_block_tensor_wr_i<N, double> &bt);

    /** \brief Fills one block of a diagonal block tensor with random values
            preserving symmetry
        \param bt Block tensor.
        \param bidx Block index in the block tensor.
     **/
    void perform(diag_block_tensor_wr_i<N, double> &bt, const index<N> &bidx);

};


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_RANDOM_H
