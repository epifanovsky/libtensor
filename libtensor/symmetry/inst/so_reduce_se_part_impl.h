#ifndef LIBTENSOR_SO_REDUCE_IMPL_PART_H
#define LIBTENSOR_SO_REDUCE_IMPL_PART_H

#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_index_subspace_builder.h>
#include "combine_part.h"

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_reduce<N, M, T>, se_part<N - M, T> >::k_clazz =
        "symmetry_operation_impl< so_reduce<N, M, T>, se_part<N - M, T> >";

template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_reduce<N, M, T>, se_part<N - M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    typedef std::pair< index<NA>, scalar_transf<T> > part_type;
    typedef std::list<part_type> plist_type;

    // Create the inverse of the mask
    mask<NA> invmsk;
    for (register size_t i = 0; i < NA; i++)
        invmsk[i] = !params.msk[i];

    params.g2.clear();
    if (params.g1.is_empty()) return;

    // Create a map of indexes and a mask of reduction steps
    sequence<NA, size_t> map(0);
    mask<NR> rsteps;
    for (register size_t i = 0, j = 0; i < NA; i++) {
        if (params.msk[i]) {
            rsteps[params.rseq[i]] = true;
            map[i] = NB + params.rseq[i];
        }
        else {
            map[i] = j++;
        }
    }

    combine_part<NA, T> cp(params.g1);
    ela_t ela(cp.get_bis(), cp.get_pdims());
    cp.perform(ela);

    dimensions<NA> bidimsa = ela.get_bis().get_block_index_dims();
    const dimensions<NA> &pdimsa = ela.get_pdims();

    // For each reduction step determine
    // - partition size (rpsz == nix)
    // - partitions in reduction range (irpa == ixa, irpb, ixb)
    index<NR> rpsz, i1rp, i2rp;
    for (register size_t i = 0; i < NR && rsteps[i]; i++) {

        register size_t j = 0;
        for (; j < NA; j++) {
            if (params.msk[j] && params.rseq[j] == i) break;
        }

        size_t j1 = j++;
        size_t p1 = pdimsa[j1];
        for (; j < NA; j++) {
            if ((! params.msk[j]) || (params.rseq[j] != i)) continue;
            if (p1 == pdimsa[j]) continue;

            size_t p2 = pdimsa[j];
            if (p2 < p1) std::swap(p1, p2);

            p1 = (p2 % p1 == 0) ? p1 : 1;
        }

        size_t nbpp = bidimsa[j1] / p1;
        rpsz[i] = p1;
        i1rp[i] = params.rblrange.get_begin()[j1] / nbpp;
        i2rp[i] = params.rblrange.get_end()[j1] / nbpp;
    }
    dimensions<NR> rpdims(index_range<NR>(i1rp, i2rp));

    // Determine the sizes of the remaining dimensions
    index<NB> i1, i2;
    index<NA> j1, j2;
    for (register size_t i = 0; i < NA; i++) {
        if (params.msk[i]) {
            j2[i] = pdimsa[i] / rpsz[params.rseq[i]] - 1;
        }
        else {
            i2[map[i]] = pdimsa[i] - 1;
        }
    }

    dimensions<NB> pdimsb(index_range<NB>(i1, i2));
    dimensions<NA> pdimsr(index_range<NA>(j1, j2));

    block_index_subspace_builder<NB, M> bb(ela.get_bis(), invmsk);
    elb_t elb(bb.get_bis(), pdimsb);

    bool empty = true;

    std::vector<bool> done(pdimsb.get_size(), false);
    // Loop over all partitions in result
    abs_index<NB> aib1(pdimsb);
    do {
        if (done[aib1.get_abs_index()]) continue;

        const index<NB> &ib1 = aib1.get_index();

        // Create a list of all unique, non-forbidden partitions from the
        // input which map on ib1
        plist_type l1;

        abs_index<NR> ai1(rpdims);
        do {
            const index<NR> &idx = ai1.get_index();

            index<NA> ia;
            for (register size_t i = 0, j = 0; i < NA; i++) {
                if (params.msk[i]) {
                    size_t ir = params.rseq[i];
                    ia[i] = (i1rp[ir] + idx[ir]) * pdimsr[i];
                }
                else {
                    ia[i] = ib1[j++];
                }
            }

            // Do not add to list if forbidden
            if (is_forbidden(ela, ia, pdimsr)) continue;

            // See if there already exist a partition in list which maps onto
            // the current
            typename plist_type::iterator il1 = l1.begin();
            for (; il1 != l1.end(); il1++) {
                if (map_exists(ela, il1->first, ia, pdimsr)) break;
            }
            // If not, just add the partition
            if (il1 == l1.end()) {
                l1.push_back(part_type(ia, scalar_transf<T>()));
            }
            // If so, add the transformation
            else {
                scalar_transf_sum<T> sum;
                sum.add(il1->second);
                sum.add(ela.get_transf(il1->first, ia));
                il1->second = sum.get_transf();
            }
        } while (ai1.inc());

        // Remove all partitions with zero transformation from list
        typename plist_type::iterator il1 = l1.begin();
        while (il1 != l1.end()) {

            if (il1->second.is_zero()) il1 = l1.erase(il1);
            else il1++;
        }

        done[aib1.get_abs_index()] = true;

        // If the list is empty the partition is automatically forbidden
        if (l1.empty()) {
            elb.mark_forbidden(ib1);
            empty = false;
            continue;
        }

        // Search maps from current index ib1
        abs_index<NB> aib2(ib1, pdimsb);
        while (aib2.inc()) {

            const index<NB> &ib2 = aib2.get_index();

            // Create a second list of partitions for ib2
            plist_type l2;

            abs_index<NR> ai2(rpdims);
            do {
                const index<NR> &idx = ai2.get_index();

                index<NA> ia;
                for (register size_t i = 0, j = 0; i < NA; i++) {
                    if (params.msk[i]) {
                        size_t ir = params.rseq[i];
                        ia[i] = (i1rp[ir] + idx[ir]) * pdimsr[i];
                    }
                    else {
                        ia[i] = ib2[j++];
                    }
                }

                // Do not add to list if forbidden
                if (is_forbidden(ela, ia, pdimsr)) continue;

                // See if there already exist a partition in list which maps onto
                // the current
                typename plist_type::iterator il2 = l2.begin();
                for (; il2 != l2.end(); il2++) {
                    if (map_exists(ela, il2->first, ia, pdimsr)) break;
                }
                // If not, just add the partition
                if (il2 == l2.end()) {
                    l2.push_back(part_type(ia, scalar_transf<T>()));
                }
                // If so, add the transformation
                else {
                    scalar_transf_sum<T> sum;
                    sum.add(il2->second);
                    sum.add(ela.get_transf(il2->first, ia));
                    il2->second = sum.get_transf();
                }
            } while (ai2.inc());

            // Remove all partitions with zero transformation from list
            typename plist_type::iterator il2 = l2.begin();
            while (il2 != l2.end()) {

                if (il2->second.is_zero()) il2 = l2.erase(il2);
                else il2++;
            }

            // If the list is empty the partition is automatically forbidden
            if (l2.empty()) {
                elb.mark_forbidden(ib2);
                done[aib2.get_abs_index()] = true;
                empty = false;
                continue;
            }

            // Check there is a map from l1 to l2
            bool found = false;
            scalar_transf<T> trb;
            il1 = l1.begin();
            for ( ; il1 != l1.end(); il1++) {

                il2 = l2.begin();
                for ( ; il2 != l2.end(); il2++) {
                    if (map_exists(ela, il1->first, il2->first, pdimsr)
                            && (il1->second == il2->second)) break;
                } // for l2

                if (il2 == l2.end()) break;

                scalar_transf<T> tr = ela.get_transf(il1->first, il2->first);
                if (found && trb != tr) break;

                trb = tr;
                found = true;
                l2.erase(il2);
            } // for l1

            if (il1 == l1.end()) {
                elb.add_map(ib1, ib2, trb);
                empty = false;
                break;
            }
        } // while aib2
    } while (aib1.inc());

    if (! empty) params.g2.insert(elb);
}

template<size_t N, size_t M, typename T>
bool
symmetry_operation_impl< so_reduce<N, M, T>, se_part<N - M, T> >::is_forbidden(
    const ela_t &el, const index<NA> &idx, const dimensions<NA> &subdims) {

    if (! el.is_forbidden(idx)) return false;

    bool forbidden = true;
    abs_index<NA> ai(subdims);
    while (ai.inc()) {
        const index<NA> &ix = ai.get_index();
        index<NA> ia;
        for (register size_t i = 0; i < NA; i++) ia[i] = idx[i] + ix[i];

        if (! el.is_forbidden(ia)) { forbidden = false; break; }
    }

    return forbidden;
}

template<size_t N, size_t M, typename T>
bool
symmetry_operation_impl< so_reduce<N, M, T>, se_part<N - M, T> >::map_exists(
    const ela_t &el, const index<NA> &i1, const index<NA> &i2,
    const dimensions<NA> &subdims) {

    if (! el.map_exists(i1, i2)) return false;

    bool exists = true;
    scalar_transf<T> tr = el.get_transf(i1, i2);

    abs_index<NA> ai(subdims);
    while (ai.inc() && exists) {
        const index<NA> &ix = ai.get_index();
        index<NA> ia1, ia2;
        for (register size_t i = 0; i < NA; i++) {
            ia1[i] = i1[i] + ix[i];
            ia2[i] = i2[i] + ix[i];
        }

        if ((! el.map_exists(ia1, ia2)) || (tr != el.get_transf(ia1, ia2))) {
            exists = false;
        }
    }

    return exists;
}

} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_SE_PART_IMPL_H
