#ifndef LIBTENSOR_SO_REDUCE_IMPL_PART_H
#define LIBTENSOR_SO_REDUCE_IMPL_PART_H

#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_index_subspace_builder.h>
#include "../combine_part.h"

namespace libtensor {

template<size_t N, size_t M, size_t NM, typename T>
const char *
symmetry_operation_impl< so_reduce<N, M, T>, se_part<NM, T> >::k_clazz =
        "symmetry_operation_impl< so_reduce<N, M, T>, se_part<NM, T> >";

template<size_t N, size_t M, size_t NM, typename T>
void
symmetry_operation_impl< so_reduce<N, M, T>, se_part<NM, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter<k_order1, T, el1_t> adapter_t;

    // Create the inverse of the mask
    mask<k_order1> invmsk;
    for (register size_t i = 0; i < k_order1; i++)
        invmsk[i] = !params.msk[i];

    params.grp2.clear();
    if (params.grp1.is_empty()) return;

    // Create a map of indexes and a mask of reduction steps
    sequence<k_order1, size_t> map(0);
    mask<k_order1> rsteps;
    for (register size_t i = 0, j = 0; i < k_order1; i++) {
        if (params.msk[i]) {
            rsteps[params.rseq[i]] = true;
            map[i] = k_order2 + params.rseq[i];
        }
        else {
            map[i] = j++;
        }
    }

    combine_part<k_order1, T> cp(params.grp1);
    el1_t el1(cp.get_bis(), cp.get_pdims());
    cp.perform(el1);

    dimensions<k_order1> bidims = el1.get_bis().get_block_index_dims();
    const dimensions<k_order1> &pdims1 = el1.get_pdims();

    // Determine the sizes of the reduced dimensions per reduction step
    // and the reduction length
    index<k_order1> ixa, ixb, nix;
    for (register size_t i = 0; i < k_order1 && rsteps[i]; i++) {
        register size_t j = 0;
        for (; j < k_order1; j++) {
            if (params.msk[j] && params.rseq[j] == i) break;
        }
        size_t j1 = j;
        nix[i] = pdims1[j1];

        for (; j < k_order1; j++) {
            if ((! params.msk[j]) || (params.rseq[j] != i)) continue;
            if (nix[i] == pdims1[j]) continue;

            size_t d1 = nix[i], d2 = pdims1[j];
            if (d1 < d2) std::swap(d1, d2);

            nix[i] = (d1 % d2 == 0) ? d2 : 1;
        }

        size_t nbpp = bidims[j1] / nix[i];
        ixa[i] = params.rblrange.get_begin()[j1] / nbpp;
        ixb[i] = params.rblrange.get_end()[j1] / nbpp;
    }

    // Determine the sizes of the remaining dimensions
    index<k_order2> ia, ib;
    index<k_order1> ja, jb;
    for (register size_t i = 0; i < k_order1; i++) {
        if (params.msk[i])
            jb[i] = pdims1[i] / nix[params.rseq[i]] - 1;
        else
            ib[map[i]] = pdims1[i] - 1;
    }

    dimensions<k_order2> pdims2(index_range<k_order2>(ia, ib));
    dimensions<k_order1> pdims1r(index_range<k_order1>(ixa, ixb));
    dimensions<k_order1> pdims1ir(index_range<k_order1>(ja, jb));

    block_index_subspace_builder<k_order2, M> bb(el1.get_bis(), invmsk);
    el2_t el2(bb.get_bis(), pdims2);

    bool empty = true;
    abs_index<k_order2> ai2a(pdims2);
    do {
        const index<k_order2> &i2a = ai2a.get_index();

        // Create a list of all possible indexes from the input
        std::list< index<k_order1> > la;
        abs_index<k_order1> ai3a(pdims1r);
        do {
            const index<k_order1> &i3a = ai3a.get_index();

            index<k_order1> i1a;
            for (register size_t i = 0, j = 0; i < k_order1; i++) {
                if (params.msk[i])
                    i1a[i] = (ixa[params.rseq[i]]
                                  + i3a[params.rseq[i]]) * pdims1ir[i];
                else
                    i1a[i] = i2a[j++];
            }

            if (! is_forbidden(el1, i1a, pdims1ir)) la.push_back(i1a);

        } while (ai3a.inc());

        if (la.empty()) {

            el2.mark_forbidden(i2a);
            empty = false;
            continue;
        }

        abs_index<k_order2> ai2b(i2a, pdims2);
        while (ai2b.inc()) {

            const index<k_order2> &i2b = ai2b.get_index();

            std::list< index<k_order1> > lb;

            abs_index<k_order1> ai3b(pdims1r);
            do {
                const index<k_order1> &i3b = ai3b.get_index();

                index<k_order1> i1b;
                for (register size_t i = 0, j = 0; i < k_order1; i++) {
                    if (params.msk[i])
                        i1b[i] = (ixa[params.rseq[i]]
                                      + i3b[params.rseq[i]]) * pdims1ir[i];
                    else
                        i1b[i] = i2b[j++];
                }

                if (! is_forbidden(el1, i1b, pdims1ir)) lb.push_back(i1b);

            } while (ai3b.inc());

            if (lb.empty()) continue;

            bool found = false;
            scalar_transf<T> tr;
            typename std::list< index<k_order1> >::iterator ila = la.begin();
            for ( ; ila != la.end(); ila++) {

                typename std::list< index<k_order1> >::iterator ilb =
                        lb.begin();
                for ( ; ilb != lb.end(); ilb++) {
                    if (map_exists(el1, *ila, *ilb, pdims1ir)) break;
                } // for lb

                if (ilb == lb.end()) break;

                scalar_transf<T> trx = el1.get_transf(*ila, *ilb);
                if (found && tr != trx) break;

                tr = trx;
                found = true;
                lb.erase(ilb);
            } // for la

            if (ila == la.end()) {
                el2.add_map(i2a, i2b, tr);
                empty = false;
                break;
            }

        } // while ai2b
    } while (ai2a.inc());

    if (! empty) params.grp2.insert(el2);
}

template<size_t N, size_t M, size_t NM, typename T>
bool symmetry_operation_impl< so_reduce<N, M, T>, se_part<NM, T> >::
is_forbidden(const el1_t &el, const index<k_order1> &idx,
        const dimensions<k_order1> &subdims) {

    if (! el.is_forbidden(idx)) return false;

    bool forbidden = true;
    abs_index<k_order1> aix(subdims);
    while (aix.inc()) {
        const index<k_order1> &ix = aix.get_index();
        index<k_order1> ia;
        for (register size_t i = 0; i < k_order1; i++) ia[i] = idx[i] + ix[i];

        if (! el.is_forbidden(ia)) { forbidden = false; break; }
    }

    return forbidden;
}

template<size_t N, size_t M, size_t NM, typename T>
bool symmetry_operation_impl< so_reduce<N, M, T>, se_part<NM, T> >::
map_exists(const el1_t &el, const index<k_order1> &ia,
        const index<k_order1> &ib, const dimensions<k_order1> &subdims) {

    if (! el.map_exists(ia, ib)) return false;

    bool exists = true;
    scalar_transf<T> tr = el.get_transf(ia, ib);

    abs_index<k_order1> aix(subdims);
    while (aix.inc() && exists) {
        const index<k_order1> &ix = aix.get_index();
        index<k_order1> i1a, i1b;
        for (register size_t i = 0; i < k_order1; i++) {
            i1a[i] = ia[i] + ix[i];
            i1b[i] = ib[i] + ix[i];
        }

        if ((! el.map_exists(i1a, i1b)) || (tr != el.get_transf(i1a, i1b))) {
            exists = false;
        }
    }

    return exists;
}

} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_IMPL_PART_H
