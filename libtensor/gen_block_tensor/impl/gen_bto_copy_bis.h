#ifndef LIBTENSOR_GEN_BTO_COPY_BIS_H
#define LIBTENSOR_GEN_BTO_COPY_BIS_H

#include <libtensor/core/block_index_space.h>

namespace libtensor {


/** \brief Computes the block index space of the result of gen_bto_copy
    \tparam N Tensor order.

    \ingroup libtensor_gen_bto
 **/
template<size_t N>
class gen_bto_copy_bis {
private:
    block_index_space<N> m_bisb; //!< Block index space of the result

public:
    gen_bto_copy_bis(
        const block_index_space<N> &bisa,
        const permutation<N> &perma) :
        m_bisb(bisa) {

        m_bisb.permute(perma);
    }

    const block_index_space<N> &get_bisb() const {
        return m_bisb;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_COPY_BIS_H
