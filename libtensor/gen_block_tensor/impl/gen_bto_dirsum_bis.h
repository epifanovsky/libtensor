#ifndef LIBTENSOR_GEN_BTO_DIRSUM_BIS_H
#define LIBTENSOR_GEN_BTO_DIRSUM_BIS_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/block_index_space.h>

namespace libtensor {


/** \brief Computes the block index space of the result of a direct sum

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M>
class gen_bto_dirsum_bis : public noncopyable {
private:
    block_index_space<N + M> m_bisc; //!< Block index space of result

public:
    /** \brief Compute the block index space
        \param bisa Block index space of A
        \param bisb Block index space of B
        \param permc Permutation of result
     **/
    gen_bto_dirsum_bis(
            const block_index_space<N> &bisa,
            const block_index_space<M> &bisb,
            const permutation<N + M> &permc);

    const block_index_space<N + M> &get_bis() const {
        return m_bisc;
    }

private:
    static dimensions<N + M> mk_dims(
            const block_index_space<N> &bisa,
            const block_index_space<M> &bisb);
};


} // namespace libtensor

#endif // LIBTENOSR_GEN_BTO_DIRSUM_BIS_H
