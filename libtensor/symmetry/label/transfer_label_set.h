#ifndef LIBTENSOR_TRANSFER_LABEL_SET_H
#define LIBTENSOR_TRANSFER_LABEL_SET_H

#include "../se_label.h"

namespace libtensor {

/** \brief Transfer all label sets from one se_label to another.

    Transfers all label sets from one se_label object to another using a map.
    The target label set can have larger dimensions than the source set.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class transfer_label_set {
public:
    static const char *k_clazz;

private:
    const se_label<N, T> &m_el;

public:
    transfer_label_set(const se_label<N, T> &el);

    template<size_t M>
    void perform(const sequence<N, size_t> &map, se_label<M, T> &to);

};

template<size_t N, typename T>
const char *transfer_label_set<N, T>::k_clazz = "transfer_label_set<N, T>";

template<size_t N, typename T>
transfer_label_set<N, T>::transfer_label_set(const se_label<N, T> &el) :
m_el(el) {

}

template<size_t N, typename T>
template<size_t M>
void transfer_label_set<N, T>::perform<M>(const sequence<N, size_t> &map,
        se_label<M, T> &to) {

#ifdef LIBTENSOR_DEBUG
    static const char *method =
            "perform<M>(const sequence<N, size_t> &, se_label<N + M, T> &)";

    static size_t k_diff = N - M;

    for (size_t i = 0; i < N; i++) {
        if (map[i] >= M) {
            throw bad_parameter(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "map");
        }
    }
#endif

    // Loop over all label sets in the source se_label and copy them
    for (typename se_label<N, T>::const_iterator iss = m_el.begin();
            iss != m_el.end(); iss++) {

        const label_set<N> &ss1 = m_el.get_subset(iss);
        const mask<N> &msk1 = ss1.get_mask();

        mask<M> msk2;
        for (register size_t k = 0; k < N; k++) {
            msk2[map[k]] = msk1[k];
        }

        label_set<M> &ss2 = to.create_subset(ss1.get_table_id());
        ss2.set_mask(msk2);

        // Copy the intrinsic labels
        for (typename label_set<N>::iterator ii = ss1.begin();
                ii != ss1.end(); ii++) {

            ss2.add_intrinsic(ss1.get_intrinsic(ii));
        }

        const dimensions<N> &bidims = ss1.get_block_index_dims();

        // Assign labels to the dimensions stemming from ss1
        for (register size_t k = 0; k < N; k++) {
            mask<M> msk;
            msk[map[k]] = true;

            size_t ktype = ss1.get_dim_type(k);
            for (size_t kpos = 0; kpos < bidims[k]; kpos++) {

                typename label_set<N>::label_t label =
                        ss1.get_label(ktype, kpos);

                if (! ss1.is_valid(label)) continue;
                ss2.assign(msk, kpos, label);
            }
        }

        ss2.match_blk_labels();
    }

}

} // namespace libtensor


#endif // LIBTENSOR_TRANSFER_LABEL_SET_H
