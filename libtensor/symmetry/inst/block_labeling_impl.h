#ifndef LIBTENSOR_BLOCK_LABELING_IMPL_H
#define LIBTENSOR_BLOCK_LABELING_IMPL_H

#include "../bad_symmetry.h"

namespace libtensor {

template<size_t N>
block_labeling<N>::block_labeling(const dimensions<N> &bidims) :
    m_bidims(bidims), m_type((size_t) -1), m_labels(0) {

    size_t cur_type = 0;
    for (size_t i = 0; i < N; i++) {
        if (m_type[i] != (size_t) -1) continue;

        m_type[i] = cur_type;
        m_labels[cur_type] =
                new blk_label_t(m_bidims[i], product_table_i::k_invalid);

        for (size_t j = i + 1; j < N; j++) {

            if (m_bidims[i] == m_bidims[j]) m_type[j] = cur_type;
        }
        cur_type++;
    }
}


template<size_t N>
void block_labeling<N>::assign(const mask<N> &msk, size_t blk, label_t l) {

    static const char *method = "assign(const mask<N> &, size_t, label_t)";

    size_t i = 0;
    for (; i < N; i++)  if (msk[i]) break;
    if (i == N) return; // mask has no true component

    size_t type = m_type[i];

#ifdef LIBTENSOR_DEBUG
    // Test if position is out of bounds
    if (blk >= m_bidims[i]) {
        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "blk");
    }

    // Test if all masked indexes are of the same type
    for (size_t j = i + 1; j < N; j++) {
        if (! msk[j]) continue;
        if (m_type[j] == type) continue;

        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "msk");
    }

#endif

    // Test if there are dimensions included in the type that are not part
    // of the mask
    bool adjust = false;
    for(i = 0; i < N; i++) {
        if (msk[i]) continue;
        if (m_type[i] == type) { adjust = true; break; }
    }

    // If yes, split dimension type into two
    size_t cur_type = type;
    if (adjust) {
        for (i = 0; i < N; i++) if (m_labels[i] == 0) break;
        cur_type = i;
        m_labels[cur_type] =
                new blk_label_t(*(m_labels[type]));

        // Assign all masked indexes to the new type.
        for (i = 0; i < N; i++) {
            if (msk[i]) m_type[i] = cur_type;
        }
    }

    // Set the new block label
    m_labels[cur_type]->at(blk) = l;
}


template<size_t N>
void block_labeling<N>::match() {

    sequence<N, size_t> types(m_type);
    sequence<N, blk_label_t*> labels(m_labels);

    for (size_t i = 0; i < N; i++) {
        m_type[i] = (size_t) -1; m_labels[i] = 0;
    }

    size_t cur_type = 0;
    for (size_t i = 0; i < N; i++) {

        size_t itype = types[i];
        if (labels[itype] == 0) continue;

        m_type[i] = cur_type;
        blk_label_t *lli = m_labels[cur_type] = labels[itype];
        labels[itype] = 0;

        for (size_t j = i + 1; j < N; j++) {
            size_t jtype = types[j];

            if (itype == jtype) {
                m_type[j] = cur_type;
                continue;
            }

            if (labels[jtype] == 0) continue;
            if (lli->size() != labels[jtype]->size()) continue;

            size_t k = 0;
            for (; k < lli->size(); k++) {
                if (lli->at(k) != labels[jtype]->at(k)) break;
            }
            if (k != lli->size()) continue;

            delete labels[jtype];
            labels[jtype] = 0;
            m_type[j] = cur_type;
            for (k = j + 1; k < N; k++) {
                if (types[k] == jtype) m_type[k] = cur_type;
            }
        }

        cur_type++;
    }
}


template<size_t N>
void block_labeling<N>::clear() {

    for (size_t i = 0; i < N && m_labels[i] != 0; i++) {
        blk_label_t &lg = *(m_labels[i]);
        for (size_t j = 0; j < lg.size(); j++)
            lg[j] = product_table_i::k_invalid;
    }

    match();
}


template<size_t N>
bool operator==(const block_labeling<N> &a, const block_labeling<N> &b) {

    if (a.get_block_index_dims() != b.get_block_index_dims()) return false;

    for (size_t i = 0; i < N; i++) {
        size_t ta = a.get_dim_type(i), tb = b.get_dim_type(i);

        for (size_t j = 0; j < a.get_dim(ta); j++) {
            if (a.get_label(ta, j) != b.get_label(tb, j)) return false;
        }
    }

    return true;
}


template<size_t N, size_t M>
void transfer_labeling(const block_labeling<N> &from,
        const sequence<N, size_t> &map, block_labeling<M> &to) {

    static const char *method =
            "transfer_labeling(const block_labeling<N> &, "
            "const sequence<N> &, block_labeling<M> &)";

#ifdef LIBTENSOR_DEBUG
    for (size_t i = 0; i < N; i++) {
        if (map[i] == (size_t) -1) continue;
        if (map[i] >= M) {
            throw bad_symmetry(g_ns, "", method,
                    __FILE__, __LINE__, "Invalid map.");
        }

        for (size_t j = i + 1; j < N; j++) {
            if (map[i] == map[j])
                throw bad_symmetry(g_ns, "", method,
                        __FILE__, __LINE__, "Invalid map.");
        }
    }
#endif

    mask<N> done;
    for (size_t i = 0; i < N; i++) {
        if (map[i] == (size_t) -1 || done[i]) continue;

        size_t typei = from.get_dim_type(i);
        mask<M> m; m[map[i]] = true;

        // Find dimensions that can be transferred together with i.
        for (size_t j = i + 1; j < N; j++) {
            if (done[j]) continue;
            if (map[j] == (size_t) -1) { done[j] = true; continue; }
            if (from.get_dim_type(j) != typei) continue;

            m[map[j]] = true;
            done[j] = true;
        }

        for (size_t j = 0; j < from.get_dim(typei); j++) {
            to.assign(m, j, from.get_label(typei, j));
        }
    }
}


} // namespace libtensor


#endif // LIBTENSOR_BLOCK_LABELING_H

