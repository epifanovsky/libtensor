#ifndef LIBTENSOR_SO_MERGE_SE_PART_IMPL_H
#define LIBTENSOR_SO_MERGE_SE_PART_IMPL_H

#include <list>
#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/core/block_index_subspace_builder.h>
#include <libtensor/core/abs_index.h>
#include "combine_part.h"

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl<so_merge<N, M, T>, se_part<N - M, T> >::k_clazz =
        "symmetry_operation_impl< so_merge<N, M, T>, se_part<N - M, T> >";

template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_merge<N, M, T>, se_part<N - M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    params.g2.clear();
    if (params.g1.is_empty()) return;

    // Determine index map N -> N - M
    mask<N> mm;
    sequence<N, size_t> map, mmap(N);
    for (register size_t i = 0, j = 0; i < N; i++) {
        if (params.msk[i]) {
            if (mmap[params.mseq[i]] != N) {
                map[i] = mmap[params.mseq[i]];
                continue;
            }

            mmap[params.mseq[i]] = j;
        }
        map[i] = j++;
        mm[i] = true;
    }

    combine_part<N, T> cp(params.g1);
    el1_t el1(cp.get_bis(), cp.get_pdims());
    cp.perform(el1);

    const dimensions<N> &pdims1 = el1.get_pdims();

    // Create result partition dimensions
    index<N - M> ia, ib;
    for (register size_t i = 0; i < N; i++) {
        if (params.msk[i] && (! mm[i])) {

            size_t d1 = (ib[map[i]] + 1), d2 = pdims1[i];
            if (d1 < d2) std::swap(d1, d2);

            ib[map[i]] = ((d1 % d2 == 0) ? d1 - 1 : 0);
        }
        else {
            ib[map[i]] = pdims1[i] - 1;
        }
    }
    dimensions<N - M> pdims2(index_range<N - M>(ia, ib));
    if (pdims2.get_size() == 1) return;

    index<N> ja, jb1, jb2;
    for (register size_t i = 0; i < N; i++) {
        if (pdims2[map[i]] == 0) {
            jb1[i] = pdims1[i];
        }
        else {
            jb2[i] = pdims2[map[i]] / pdims1[i] - 1;
        }
    }
    dimensions<N> pdims1s1(index_range<N>(ja, jb1));
    dimensions<N> pdims1s2(index_range<N>(ja, jb2));

    block_index_subspace_builder<N - M, M> bb(el1.get_bis(), mm);
    el2_t el2(bb.get_bis(), pdims2);

    // Merge the partitions
    abs_index<N - M> ai(pdims2);
    do {
        register size_t i;

        const index<N - M> &i2a = ai.get_index();
        // Create pre-merge index
        index<N> i1a;
        for (i = 0; i < N; i++) {
            i1a[i] = i2a[map[i]] / pdims1s2[i];
        }

        // Check if index forbidden
        if (is_forbidden(el1, i1a, pdims1s1)) {
            el2.mark_forbidden(i2a);
            continue;
        }

        bool found = false;
        index<N> i1b = el1.get_direct_map(i1a);
        while (! found && i1a < i1b) {
            // Check if i1b can be converted into a proper result index
            for (i = 0; i < N; i++) {
                if (! params.msk[i]) continue;

                size_t j = i + 1;
                for (; j < N; j++) {
                    if (map[i] != map[j]) continue;

                    if (i1b[i] * pdims1s2[i] != i1b[j] * pdims1s2[j]) break;
                    if (i1b[i] % pdims1s1[i] != 0 ||
                            i1b[j] % pdims1s1[j] != 0) break;
                }
                if (j != N) break;
            }
            if (i == N) found = true;
            else i1b = el1.get_direct_map(i1b);
        }
        if (! found) continue;

        if (map_exists(el1, i1a, i1b, pdims1s1)) {

            index<N - M> i2b;
            for (i = 0; i < N; i++) i2b[map[i]] = i1b[i] / pdims1s1[i];

            el2.add_map(i2a, i2b, el1.get_transf(i1a, i1b));
        }

    } while (ai.inc());

    params.g2.insert(el2);
}


template<size_t N, size_t M, typename T>
bool symmetry_operation_impl< so_merge<N, M, T>, se_part<N - M, T> >::
is_forbidden(const el1_t &el, const index<N> &idx,
        const dimensions<N> &subdims) {

    if (! el.is_forbidden(idx)) return false;

    bool forbidden = true;
    abs_index<N> aix(subdims);
    while (aix.inc()) {
        const index<N> &ix = aix.get_index();
        index<N> ia;
        for (register size_t i = 0; i < N; i++)
            ia[i] = idx[i] + ix[i];

        if (! el.is_forbidden(ia)) { forbidden = false; break; }
    }

    return forbidden;
}


template<size_t N, size_t M, typename T>
bool symmetry_operation_impl< so_merge<N, M, T>, se_part<N - M, T> >::
map_exists(const el1_t &el, const index<N> &ia,
        const index<N> &ib, const dimensions<N> &subdims) {

    if (! el.map_exists(ia, ib)) return false;

    bool exists = true;
    scalar_transf<T> tr = el.get_transf(ia, ib);

    abs_index<N> aix(subdims);
    while (aix.inc() && exists) {
        const index<N> &ix = aix.get_index();
        index<N> i1a, i1b;
        for (register size_t i = 0; i < N; i++) {
            i1a[i] = ia[i] + ix[i];
            i1b[i] = ib[i] + ix[i];
        }

        if ((! el.map_exists(i1a, i1b)) ||
                (tr != el.get_transf(i1a, i1b))) {
            exists = false;
        }
    }

    return exists;
}


} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_PERM_H
