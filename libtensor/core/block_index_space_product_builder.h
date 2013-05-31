#ifndef LIBTENSOR_BLOCK_INDEX_SPACE_PRODUCT_BUILDER_H
#define LIBTENSOR_BLOCK_INDEX_SPACE_PRODUCT_BUILDER_H

#include "block_index_space.h"

namespace libtensor {


/** \brief Builds the direct product of two block %index spaces.
    \tparam N Order of the first space.
    \tparam M Order of the second space.

    Constructs a block %index space from two smaller block %index
    spaces by concatenating the dimensions.

    \ingroup libtensor_core
 **/
template<size_t N, size_t M>
class block_index_space_product_builder {
public:
    static const char *k_clazz; //!< Class name

private:
    block_index_space<N + M> m_bis; //!< Result

public:
    /** \brief Constructs the subspace using two block index spaces
            and a %permutation.
     **/
    block_index_space_product_builder(const block_index_space<N> &bisa,
            const block_index_space<M> &bisb, const permutation<N + M>&perm);

    /** \brief Returns the subspace
     **/
    const block_index_space<N + M> &get_bis() const {
        return m_bis;
    }

private:
    static dimensions<N + M> make_dims(const block_index_space<N> &bisa,
            const block_index_space<M> &bisb);

};


template<size_t N, size_t M>
const char *block_index_space_product_builder<N, M>::k_clazz =
    "block_index_space_product_builder<N, M>";


template<size_t N, size_t M>
block_index_space_product_builder<N, M>::block_index_space_product_builder(
    const block_index_space<N> &bisa, const block_index_space<M> &bisb,
    const permutation<N + M> &perm) :

    m_bis(make_dims(bisa, bisb)) {

    mask<N> mska, mska1;
    mask<M> mskb, mskb1;
    mask<N + M> mskx;
    bool done;
    size_t i;

    i = 0;
    done = false;
    while(!done) {
        while(i < N && mska[i]) i++;
        if(i == N) {
            done = true;
            continue;
        }

        size_t typ = bisa.get_type(i);
        for(size_t j = 0; j < N; j++) {
            mskx[j] = mska1[j] = bisa.get_type(j) == typ;
        }
        const split_points &pts = bisa.get_splits(typ);
        for(size_t j = 0; j < pts.get_num_points(); j++)
            m_bis.split(mskx, pts[j]);

        mska |= mska1;
    }
    for(size_t j = 0; j < N; j++) mskx[j] = false;

    i = 0;
    done = false;
    while(!done) {
        while(i < M && mskb[i]) i++;
        if(i == M) {
            done = true;
            continue;
        }

        size_t typ = bisb.get_type(i);
        for(size_t j = 0; j < M; j++) {
            mskx[N + j] = mskb1[j] =
                bisb.get_type(j) == typ;
        }
        const split_points &pts = bisb.get_splits(typ);
        for(size_t j = 0; j < pts.get_num_points(); j++)
            m_bis.split(mskx, pts[j]);

        mskb |= mskb1;
    }

    m_bis.match_splits();
    m_bis.permute(perm);

}


template<size_t N, size_t M>
dimensions<N + M> block_index_space_product_builder<N, M>::make_dims(
    const block_index_space<N> &bisa, const block_index_space<M> &bisb) {

    static const char *method =
        "make_dims(const block_index_space<N>&, "
        "const block_index_space<M>&, const permutation<N + M> &)";

    index<N + M> i1, i2;
    const dimensions<N> &dimsa = bisa.get_dims();
    const dimensions<M> &dimsb = bisb.get_dims();

    for(size_t i = 0; i < N; i++) {
        i2[i] = dimsa[i] - 1;
    }
    for(size_t i = 0; i < M; i++) {
        i2[i + N] = dimsb[i] - 1;
    }

    return dimensions<N + M>(index_range<N + M>(i1, i2));
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_INDEX_COMBINED_SPACE_BUILDER_H
