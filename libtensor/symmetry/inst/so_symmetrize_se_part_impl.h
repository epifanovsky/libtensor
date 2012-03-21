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

    std::vector<size_t> map;

    register size_t i = 0;
    for (; i < N && ! params.msk[i]; i++) ;
    size_t idim = pdims[i];
    map.push_back(i++);

    for (; i < N; i++) {
        if (! params.msk[i]) continue;
        if (pdims[i] != idim) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Incompatible dimensions.");
        }
        map.push_back(i);
    }

    se_part<N, T> sp1(cp.get_bis(), pdims);
    cp.perform(sp1);

    se_part<N, T> sp2(cp.get_bis(), pdims);
    abs_index<N> ai(pdims);
    do {

        const index<N> &i1 = ai.get_index();
        for (i = 1; i < map.size(); i++) {
            if (i1[map[i - 1]] > i1[map[i]]) break;
        }
        if (i != map.size()) continue;

        if (is_forbidden(sp1, i1, map)) {
            mark_forbidden(sp2, i1, map);
            continue;
        }

        index<N> i2 = sp1.get_direct_map(i1);
        bool found = false;
        while (!found && i1 < i2) {
            if (map_exists(sp1, i1, i2, map, params.symm)) found = true;
            else i2 = sp1.get_direct_map(i2);
        }
        if (found)
            add_map(sp2, i1, i2, sp1.get_sign(i1, i2), map, params.symm);

    } while (ai.inc());

    params.grp2.insert(sp2);
}

template<size_t N, typename T>
bool
symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::is_forbidden(
        const se_part<N, T> &sp, const index<N> &i1,
        const std::vector<size_t> &map) {

    index<N> ix(i1);
    permutation_generator pg(map.size());
    do {
        for (register size_t i = 0; i < map.size(); i++)
            ix[map[i]] = i1[map[pg[i]]];
        if (! sp.is_forbidden(ix)) return false;
    } while (pg.next());

    return true;
}

template<size_t N, typename T>
void
symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::mark_forbidden(
        se_part<N, T> &sp, const index<N> &i1,
        const std::vector<size_t> &map) {

    index<N> ix(i1);
    permutation_generator pg(map.size());
    do {
        for (register size_t i = 0; i < map.size(); i++)
            ix[map[i]] = i1[map[pg[i]]];
        sp.mark_forbidden(ix);
    } while (pg.next());
}

template<size_t N, typename T>
bool symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::map_exists(
        const se_part<N, T> &sp, const index<N> &i1, const index<N> &i2,
        const std::vector<size_t> &map, bool symm) {

    index<N> j1(i1), j2(i2);
    permutation_generator pg(map.size());
    size_t n = 0;
    bool sign;
    do {
        for (register size_t i = 0; i < map.size(); i++) {
            j1[map[i]] = i1[map[pg[i]]];
            j2[map[i]] = i2[map[pg[i]]];
        }
        if (sp.map_exists(j1, j2)) {
            sign = sp.get_sign(j1, j2);
            if (n % 2 != 0 && ! symm) sign = ! sign;
            break;
        }
        else if ((! sp.is_forbidden(j1)) || (! sp.is_forbidden(j2))) {
            return false;
        }
        n++;
    } while (pg.next());

     do {
         for (register size_t i = 0; i < map.size(); i++) {
             j1[map[i]] = i1[map[pg[i]]];
             j2[map[i]] = i2[map[pg[i]]];
         }
         if (sp.map_exists(j1, j2)) {
            bool signx = sp.get_sign(j1, j2);
            if ((sign == signx && (n % 2 != 0 && ! symm)) ||
                    (sign != signx && (symm || n % 2 == 0))) return false;
        }
        else if ((! sp.is_forbidden(j1)) || (! sp.is_forbidden(j2))) {
            return false;
        }

        n++;
    } while (pg.next());

    return true;
}


template<size_t N, typename T>
void symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::add_map(
        se_part<N, T> &sp, const index<N> &i1, const index<N> &i2,
        bool sign, const std::vector<size_t> &map, bool symm) {

    index<N> j1(i1), j2(i2);
    permutation_generator pg(map.size());
    size_t n = 0;
    do {
        for (register size_t i = 0; i < map.size(); i++) {
            j1[map[i]] = i1[map[pg[i]]];
            j2[map[i]] = i2[map[pg[i]]];
        }
        sp.add_map(j1, j2, (n % 2 == 0 || symm ? sign : ! sign));
        n++;
    } while (pg.next());
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_PART_IMPL_H
