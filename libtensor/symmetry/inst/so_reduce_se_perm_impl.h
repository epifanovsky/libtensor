#ifndef LIBTENSOR_SO_REDUCE_SE_PERM_IMPL_H
#define LIBTENSOR_SO_REDUCE_SE_PERM_IMPL_H

#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/core/permutation_builder.h>
#include "../permutation_group.h"

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_reduce<N, M, T>, se_perm<N - M, T> >::k_clazz =
        "symmetry_operation_impl< so_reduce<N, M, T>, se_perm<N - M, T> >";

template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_reduce<N, M, T>, se_perm<N - M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(symmetry_operation_params_t&)";

    //  Adapter type for the input group
    typedef symmetry_element_set_adapter<k_order1, T, el1_t> adapter_t;

    adapter_t adapter1(params.grp1);
    permutation_group<k_order1, T> grp1(adapter1);
    permutation_group<k_order1, T> grp2;

    sequence<k_order1, size_t> seq(0);
    for (register size_t i = 0; i < k_order1; i++) {
        if (params.msk[i]) seq[i] = params.rseq[i] + 1;
    }
    grp1.stabilize(seq, grp2);

    symmetry_element_set<k_order1, T> set(el1_t::k_sym_type);
    grp2.convert(set);

    adapter_t g2(set);
    params.grp2.clear();

    const index<k_order1> &bia = params.rblrange.get_begin();
    const index<k_order1> &bib = params.rblrange.get_end();
    const index<k_order1> &ia = params.riblrange.get_begin();
    const index<k_order1> &ib = params.riblrange.get_end();

    for (typename adapter_t::iterator it = g2.begin(); it != g2.end(); it++) {

        const el1_t &e2 = g2.get_elem(it);

        // Exclude all permutations for which the block indexes and in-block
        // indexes of the reduction range differ
        index<k_order1> bia2(bia), bib2(bib), ia2(ia), ib2(ib);
        bia2.permute(e2.get_perm());
        bib2.permute(e2.get_perm());
        ia2.permute(e2.get_perm());
        ib2.permute(e2.get_perm());

        register size_t j = 0;
        for (; j < k_order1; j++) {
            if (! params.msk[j]) continue;
            if (bia2[j] != bia[j] || bib2[j] != bib[j] ||
                    ia2[j] != ia[j] || ib2[j] != ib[j]) break;
        }
        if (j != k_order1) continue;

        sequence<k_order1, size_t> seq1a(0), seq2a(0);
        sequence<k_order2, size_t> seq1b(0), seq2b(0);
        for (j = 0; j < k_order1; j++) seq1a[j] = seq2a[j] = j;
        e2.get_perm().apply(seq2a);

        for (register size_t j = 0, k = 0; j < k_order1; j++) {
            if (params.msk[j]) continue;

            seq1b[k] = seq1a[j];
            seq2b[k] = seq2a[j];
            k++;
        }

        permutation_builder<k_order2> pb(seq2b, seq1b);

        if (pb.get_perm().is_identity()) {
            if (e2.get_transf().is_identity()) continue;

            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Anti-symmetric identity permutation.");
        }

        params.grp2.insert(el2_t(pb.get_perm(), e2.get_transf()));
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_SE_PERM_IMPL_H
