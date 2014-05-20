#ifndef LIBTENSOR_SO_SYMMETRIZE_SE_PERM_IMPL_H
#define LIBTENSOR_SO_SYMMETRIZE_SE_PERM_IMPL_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/core/permutation_generator.h>
#include "../permutation_group.h"

namespace libtensor {


template<size_t N, typename T>
const char *
symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >::k_clazz =
        "symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;
    typedef std::pair< permutation<N>, scalar_transf<T> > gen_perm_t;
    typedef std::list<gen_perm_t> perm_list_t;

    // Number of groups and number of indexes per group
    size_t ngrp = 0, nidx = 0;
    for (register size_t i = 0; i < N; i++) {
        if (params.idxgrp[i] == 0) continue;
        ngrp = std::max(ngrp, params.idxgrp[i]);
        nidx = std::max(nidx, params.symidx[i]);
    }

    // Create map and msk
    sequence<N, size_t> map(0);
    mask<N> msk;
    for (register size_t i = 0; i < N; i++) {
        if (params.idxgrp[i] == 0) { msk[i] = true; continue; }
        map[(params.idxgrp[i] - 1) * nidx + params.symidx[i] - 1] = i;
        msk[i] = (params.symidx[i] != 1);
    }

    // Create pair and cyclic permutations
    permutation<N> pp, cp;
    for (size_t k = 1; k < ngrp; k++) {
        for (size_t i = 1 ; i <= nidx; i++) {
            register size_t j = 0;
            for (; j < N; j++) {
                if ((params.idxgrp[j] == k) &&
                        (params.symidx[j] == i)) break;
            }
            size_t i1 = j;
            for (j = 0; j < N; j++) {
                if ((params.idxgrp[j] == (k + 1)) &&
                        (params.symidx[j] == i)) break;
            }
            if (k == 1) pp.permute(i1, j);
            cp.permute(i1, j);
        }
    }

    if (params.g1.is_empty()) {
        params.g2.clear();
        if (ngrp > 2) params.g2.insert(se_perm<N, T>(cp, params.trc));
        params.g2.insert(se_perm<N, T>(pp, params.trp));
        return;
    }

    // Create a list of all permutations in the group
    adapter_t ad(params.g1);
    perm_list_t l0, l1, l2;
    std::set<size_t> done;
    done.insert(0);
    for (typename adapter_t::iterator it = ad.begin(); it != ad.end(); it++) {

        const element_t &e1 = ad.get_elem(it);

        size_t idx = encode(e1.get_perm());
        if (done.count(idx) != 0) continue;
        done.insert(idx);
        l0.push_back(gen_perm_t(e1.get_perm(), e1.get_transf()));
        l1.push_back(gen_perm_t(e1.get_perm(), e1.get_transf()));
    }

    while (! l1.empty()) {
        const gen_perm_t &cur = l1.front();
        for (typename perm_list_t::iterator il = l0.begin();
                il != l0.end(); il++) {

            permutation<N> p(cur.first);
            p.permute(il->first);
            size_t idx = encode(p);
            if (done.count(idx) != 0) continue;
            done.insert(idx);

            scalar_transf<T> tr(cur.second);
            tr.transform(il->second);
            l1.push_back(gen_perm_t(p, tr));
        }
        l2.push_back(cur);
        l1.pop_front();
    }

    // Loop through all symmetrizations and check which permutations survive
    permutation_generator<N> pg(msk);

    scalar_transf<T> trx;
    permutation<N> px, p1;
    while (pg.next()) {
        const permutation<N> &p2 = pg.get_perm();

        if (ngrp == 2) {
            trx.transform(params.trp);
            px.permute(pp);
        }
        else {
            size_t i = 0;
            for (; i < N && p2[i] == p1[i]; i++) {
                trx.transform(params.trc); px.permute(cp);
            }
            trx.transform(params.trp);
            px.permute(pp);
            i++;
            for (; i < N; i++) {
                trx.transform(params.trc); px.permute(cp);
            }
        }

        permutation<N> pxinv(px, true);
        scalar_transf<T> trxinv(trx); trxinv.invert();

        // Copy p2 to p1
        p1.reset();
        p1.permute(p2);

        // Permute generating set and create a new permutation group
        permutation_group<N, T> grpx;
        for (typename perm_list_t::iterator it = l0.begin();
                it != l0.end(); it++) {

            permutation<N> pi(pxinv);
            pi.permute(it->first).permute(px);
            scalar_transf<T> ti(trxinv);
            ti.transform(it->second).transform(trx);

            grpx.add_orbit(ti, pi);
        }

        // Test permutations in list against new permutation group
        typename perm_list_t::iterator it = l2.begin();
        while (it != l2.end()) {
            if (! grpx.is_member(it->second, it->first)) it = l2.erase(it);
            else it++;
        }
    }

    // At last, add all elements remaining in the list to group2
    permutation_group<N, T> group2;
    for (typename perm_list_t::iterator it = l2.begin();
            it != l2.end(); it++) {
        group2.add_orbit(it->second, it->first);
    }

    if (ngrp > 2) group2.add_orbit(params.trc, cp);
    group2.add_orbit(params.trp, pp);

    params.g2.clear();
    group2.convert(params.g2);
}


template<size_t N, typename T>
size_t symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >::encode(
        const permutation<N> &p) {

    permutation<N> pinv(p, true);
    size_t idx = 0;
    for (register size_t i = 0, j = N; i != N - 1; i++, j--) {
        size_t ii = 0;
        for (register size_t k = 0; k < pinv[i]; k++) {
            if (p[k] > i) ii++;
        }
        idx = idx * j + ii;
    }

    return idx;
}


} // namespace libtensor


#endif // LIBTENSOR_SO_SYMMETRIZE_SE_PERM_IMPL_H
