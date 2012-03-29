#ifndef LIBTENSOR_SO_MERGE_SE_PERM_IMPL_H
#define LIBTENSOR_SO_MERGE_SE_PERM_IMPL_H

#include <libtensor/exception.h>
#include <libtensor/core/permutation_builder.h>
#include "../bad_symmetry.h"
#include "../permutation_group.h"

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_merge<N, M, T>, se_perm<N - M, T> >::k_clazz =
        "symmetry_operation_impl< so_merge<N, M, T>, se_perm<N - M, T> >";

template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_merge<N, M, T>, se_perm<N - M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(symmetry_operation_params_t&)";

    //  Adapter type for the input group
    typedef symmetry_element_set_adapter<k_order1, T, el1_t> adapter1_t;

    // Special case for N - M == 1
    if (k_order2 == 1) {
        adapter1_t g1(params.grp1);
        params.grp2.clear();
        for (typename adapter1_t::iterator it = g1.begin();
                it != g1.end(); it++) {

            const el1_t &e1 = g1.get_elem(it);
            if (! e1.get_transf().is_identity()) {
                throw bad_symmetry(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "perm + transf.");
            }
        }
        return;
    }

    // Create the stabilizing sequence for permutation group
    mask<k_order1> msteps, mm;
    sequence<k_order1, size_t> sseq(0);
    for (register size_t i = 0; i < k_order1; i++) {
        if (! params.msk[i]) continue;

        if (msteps[params.mseq[i]])
            mm[i] = true;
        else
            msteps[params.mseq[i]] = true;

        sseq[i] = params.mseq[i] + 1;
    }

    params.grp2.clear();
    adapter1_t g1(params.grp1);
    permutation_group<k_order1, T> grp1(g1);
    permutation_group<k_order1, T> grp2;
    grp1.stabilize(sseq, grp2);

    symmetry_element_set<k_order1, T> set(el1_t::k_sym_type);
    grp2.convert(set);

    adapter1_t g2(set);
    for (typename adapter1_t::iterator it = g2.begin(); it != g2.end(); it++) {

        const el1_t &e2 = g2.get_elem(it);

        sequence<k_order1, size_t> seq1a(0), seq2a(0);
        sequence<k_order2, size_t> seq1b(0), seq2b(0);

        for (register size_t i = 0; i < k_order1; i++)
            seq1a[i] = seq2a[i] = i;
        e2.get_perm().apply(seq2a);

        for (size_t i = 0, j = 0; i < k_order1; i++) {
            if (mm[i]) continue;

            size_t ii = seq2a[i];
            if (params.msk[i]) {
                for (size_t k = 0; k < k_order1; k++) {
                    if (! params.msk[k] || params.mseq[k] != params.mseq[i])
                        continue;

                    ii = std::min(ii, seq2a[k]);
                }
            }

            seq1b[j] = seq1a[i];
            seq2b[j] = ii;
            j++;
        }

        permutation_builder<k_order2> pb(seq2b, seq1b);
        if (pb.get_perm().is_identity()) {
            if (e2.get_transf().is_identity()) continue;

            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "perm + transf.");
        }

        params.grp2.insert(el2_t(pb.get_perm(), e2.get_transf()));
    }
}




} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_PERM_IMPL_H
