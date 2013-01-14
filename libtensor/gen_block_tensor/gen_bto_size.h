#ifndef LIBTENSOR_GEN_BTO_SIZE_H
#define LIBTENSOR_GEN_BTO_SIZE_H

#include <libtensor/core/noncopyable.h>

namespace libtensor {


template<size_t N, typename Traits>
class gen_bto_size : public noncopyable {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

public:
    /** \brief Returns the size of a block tensor
        \param bta Output block tensor.
     **/
    size_t get_size(gen_block_tensor_rd_i<N, bti_traits> &bt);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SIZE_H
