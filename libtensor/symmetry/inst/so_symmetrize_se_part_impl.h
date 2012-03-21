#ifndef LIBTENSOR_SO_SYMMETRIZE_SE_PART_IMPL_H
#define LIBTENSOR_SO_SYMMETRIZE_SE_PART_IMPL_H

#include <libtensor/defs.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/permutation_generator.h>
#include "../bad_symmetry.h"
#include "../combine_part.h"

namespace libtensor {

template<size_t N, typename T>
const char *symmetry_operation_impl< so_symmetrize<N, T>,
se_part<N, T> >::k_clazz =
        "symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >";

template<size_t N, typename T>
void symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    combine_part<N, T> cp(params.grp1);
    const dimensions<N> &pdims = cp.get_pdims();

    size_t ngrp = 0, nidx = 0;
    for (register size_t i = 0; i < N; i++) {
        if (params.idxgrp[i] == 0) continue;

        ngrp = std::max(ngrp, params.idxgrp[i]);
        nidx = std::max(nidx, params.symidx[i]);
    }
    std::vector< std::vector<size_t> > map(ngrp, std::vector<size_t>(nidx, 0));

    for (register size_t i = 0; i < N; i++) {
        if (params.idxgrp[i] == 0) continue;

        map[params.idxgrp[i] - 1][params.symidx[i] - 1] = i;
    }

#ifdef LIBTENSOR_DEBUG
    for (register size_t i = 1; i < ngrp; i++) {
        for (register size_t j = 0; j < nidx; j++) {
            if (pdims[map[i][j]] != pdims[map[0][j]]) {
                throw bad_symmetry(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Incompatible dimensions.");
            }
        }
    }
#endif // LIBTENSOR_DEBUG

    // A B + sP A B
    // C D      C D

    // A+sA' B+sC'
    // C+sB' D+sD'

    // A -> tB, A'-> tA'
    se_part<N, T> sp1(cp.get_bis(), pdims);
    cp.perform(sp1);

    se_part<N, T> sp2(cp.get_bis(), pdims);
    abs_index<N> ai(pdims);
    do {

        const index<N> &i1 = ai.get_index();

        if (is_forbidden(sp1, i1, map)) {
            mark_forbidden(sp2, i1, map);
            continue;
        }

        index<N> i2 = sp1.get_direct_map(i1);
        bool found = false;
        while (!found && i1 < i2) {
            if (map_exists(sp1, i1, i2, map)) found = true;
            else i2 = sp1.get_direct_map(i2);
        }

        if (found)
            add_map(sp2, i1, i2, sp1.get_sign(i1, i2), map);

    } while (ai.inc());

    params.grp2.insert(sp2);
}

template<size_t N, typename T>
bool
symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::is_forbidden(
        const se_part<N, T> &sp, const index<N> &i1, const map_t &map) {

    index<N> ix(i1);
    permutation_generator pg(map.size());
    do {
        for (register size_t i = 0; i < pg.size(); i++) {
            const std::vector<size_t> &mx = map[i], &my = map[pg[i]];
            for (register size_t j = 0; j < mx.size(); j++)
                ix[mx[j]] = i1[my[j]];
        }
        if (! sp.is_forbidden(ix)) return false;
    } while (pg.next());

    return true;
}

template<size_t N, typename T>
void
symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::mark_forbidden(
        se_part<N, T> &sp, const index<N> &i1, const map_t &map) {

    index<N> ix(i1);
    permutation_generator pg(map.size());
    do {
        for (register size_t i = 0; i < pg.size(); i++) {
            const std::vector<size_t> &mx = map[i], &my = map[pg[i]];
            for (register size_t j = 0; j < mx.size(); j++)
                ix[mx[j]] = i1[my[j]];
        }
        sp.mark_forbidden(ix);
    } while (pg.next());
}

template<size_t N, typename T>
bool symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::map_exists(
        const se_part<N, T> &sp, const index<N> &i1, const index<N> &i2,
        const map_t &map) {

    index<N> j1(i1), j2(i2);
    permutation_generator pg(map.size());
    bool sign;
    do {
        for (register size_t i = 0; i < pg.size(); i++) {
            const std::vector<size_t> &mx = map[i], &my = map[pg[i]];
            for (register size_t j = 0; j < mx.size(); j++) {
                j1[mx[j]] = i1[my[j]];
                j2[mx[j]] = i2[my[j]];
            }
        }

        if (sp.map_exists(j1, j2)) {
            sign = sp.get_sign(j1, j2);
            break;
        }
        else if ((! sp.is_forbidden(j1)) || (! sp.is_forbidden(j2))) {
            return false;
        }
    } while (pg.next());

    while (pg.next()) {
        for (register size_t i = 0; i < map.size(); i++) {
            const std::vector<size_t> &mx = map[i], &my = map[pg[i]];
            for (register size_t j = 0; j < mx.size(); j++) {
                j1[mx[j]] = i1[my[j]];
                j2[mx[j]] = i2[my[j]];
            }
        }
        if (sp.map_exists(j1, j2)) {
            if (sign != sp.get_sign(j1, j2)) return false;
        }
        else if ((! sp.is_forbidden(j1)) || (! sp.is_forbidden(j2))) {
            return false;
        }
    }

    return true;
}


template<size_t N, typename T>
void symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::add_map(
        se_part<N, T> &sp, const index<N> &i1, const index<N> &i2,
        bool sign, const map_t &map) {

    index<N> j1(i1), j2(i2);
    permutation_generator pg(map.size());
    do {
        for (register size_t i = 0; i < map.size(); i++) {
            const std::vector<size_t> &mx = map[i], &my = map[pg[i]];
            for (register size_t j = 0; j < mx.size(); j++) {
                j1[mx[j]] = i1[my[j]];
                j2[mx[j]] = i2[my[j]];
            }
        }
        sp.add_map(j1, j2, sign);
    } while (pg.next());
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_PART_IMPL_H
