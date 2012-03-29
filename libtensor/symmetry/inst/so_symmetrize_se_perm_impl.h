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

    typedef se_perm<2 * N, T> element2_t;
    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;
    typedef symmetry_element_set_adapter<2 * N, T, element2_t> adapter2_t;

    size_t ngrp = 0, nidx = 0;
    for (register size_t i = 0; i < N; i++) {
        if (params.idxgrp[i] == 0) continue;
        ngrp = std::max(ngrp, params.idxgrp[i]);
        nidx = std::max(nidx, params.symidx[i]);
    }

    std::vector<std::vector<size_t> > map(ngrp, std::vector<size_t>(nidx, 0));
    for (register size_t i = 0; i < N; i++) {
        if (params.idxgrp[i] == 0) continue;
        map[params.idxgrp[i] - 1][params.symidx[i] - 1] = i;
    }

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
                if ((params.idxgrp[j] == k + 1) &&
                        (params.symidx[j] == i)) break;
            }
            if (k == 1) pp.permute(i1, j);
            cp.permute(i1, j);
        }
    }
    // Permutation
    // 0123->1230
    // 1320->2031
    // 0123->2310


    // Current symmetrization
    // 0123->1320

    symmetry_element_set<N, T> set(element_t::k_sym_type);
    for (typename symmetry_element_set<N, T>::const_iterator it =
            params.grp1.begin(); it != params.grp1.end(); it++) {
        set.insert(params.grp1.get_elem(it));
    }

    sequence<2 * N, size_t> stabseq;
    for (register size_t i = 0; i < N; i++) {
        stabseq[i] = stabseq[i + N] = i + 1;
    }

    adapter_t adapter(params.grp1);
    permutation_generator pg(ngrp);
    while (pg.next()) {
        // Create permutation group for direct sum of A and P A
        permutation_group<2 * N, T> grpx;

        // Loop over set and add all symmetric elements to px
        adapter_t ad1(set);
        for (typename adapter_t::iterator it = ad1.begin();
                it != ad1.end(); it++) {

            const element_t &sp = ad1.get_elem(it);
            if (sp.is_symm()) {
                sequence<N, size_t> seq;
                sequence<2 * N, size_t> seq1a, seq2a;
                for (register size_t i = 0, j = N; i < N; i++, j++) {
                    seq1a[i] = seq2a[i] = seq[i] = i;
                    seq1a[j] = seq2a[j] = j;
                }
                sp.get_perm().apply(seq);
                for (register size_t i = 0; i < N; i++) seq2a[i] = seq[i];

                permutation_builder<2 * N> pb(seq2a, seq1a);
                grpx.add_orbit(true, pb.get_perm());
            }
            else {
                sequence<N, size_t> seq;
                sequence<2 * N, size_t> seq1a, seq2a;
                for (register size_t i = 0, j = N; i < N; i++, j++) {
                    seq1a[i] = seq2a[i] = seq[i] = i;
                    seq1a[j] = seq2a[j] = j;
                }
                sp.get_perm().apply(seq);
                for (register size_t i = 0; i < N; i++) seq2a[i] = seq[i];

                for (typename adapter_t::iterator itx = adapter.begin();
                        itx != adapter.end(); itx++) {

                    const element_t &spx = adapter.get_elem(itx);
                    if (spx.is_symm()) continue;

                    // Create the rest of seq2a
                    for (register size_t i = 0; i < N; i++) seq[i] = i;
                    spx.get_perm().apply(seq);
                    for (register size_t i = 0; i < N; i++)
                        seq2a[i + N] = seq[i] + N;
                    for (register size_t i = 0; i < ngrp; i++) {
                        const std::vector<size_t> &mx = map[i];
                        const std::vector<size_t> &my = map[pg[i]];
                        for (register size_t j = 0; j < nidx; j++) {
                            seq1a[mx[j] + N] = my[j] + N;
                            seq2a[mx[j] + N] = seq[my[j]] + N;
                        }
                    }
                    permutation_builder<2 * N> pb(seq2a, seq1a);
                    grpx.add_orbit(false, pb.get_perm());
                }
            }
        }

        for (typename adapter_t::iterator it = adapter.begin();
                it != adapter.end(); it++) {

            const element_t &sp = adapter.get_elem(it);
            if (! sp.is_symm()) continue;

            // Create seq2a
            sequence<N, size_t> seq;
            sequence<2 * N, size_t> seq1a, seq2a;
            for (register size_t i = 0, j = N; i < N; i++, j++) {
                seq1a[i] = seq2a[i] = seq[i] = i;
                seq1a[j] = seq2a[j] = j;
            }
            sp.get_perm().apply(seq);
            for (register size_t i = 0; i < N; i++)
                seq2a[i + N] = seq[i] + N;
            for (register size_t i = 0; i < ngrp; i++) {
                const std::vector<size_t> &mx = map[i];
                const std::vector<size_t> &my = map[pg[i]];
                for (register size_t j = 0; j < nidx; j++) {
                    seq1a[mx[j] + N] = my[j] + N;
                    seq2a[mx[j] + N] = seq[my[j]] + N;
                }
            }
            permutation_builder<2 * N> pb(seq2a, seq1a);
            grpx.add_orbit(true, pb.get_perm());
        }

        // Stabilize px
        permutation_group<2 * N, T> grpy;
        grpx.stabilize(stabseq, grpy);

        // Convert px to symmetry element set
        symmetry_element_set<2 * N, T> set2(element2_t::k_sym_type);
        grpy.convert(set2);

        // Loop over set and project down
        adapter2_t ad2(set2);
        set.clear();
        for (typename adapter2_t::iterator it = ad2.begin();
                it != ad2.end(); it++) {

            const element2_t &sp = ad2.get_elem(it);
            sequence<2 * N, size_t> seq;
            sequence<N, size_t> seq1b, seq2b;
            for (register size_t i = 0, j = N; i < N; i++, j++) {
                seq1b[i] = seq2b[i] = seq[i] = i;
                seq[j] = j;
            }
            sp.get_perm().apply(seq);
            for (register size_t i = 0; i < N; i++)
                seq2b[i] = std::min(seq[i], seq[i + N]);

            permutation_builder<N> pb(seq2b, seq1b);
            if (pb.get_perm().is_identity()) continue;
            set.insert(element_t(pb.get_perm(), sp.is_symm()));
        }
    }

    adapter_t adapter2(set);
    permutation_group<N, T> grp2(adapter2);

    if (ngrp > 2) grp2.add_orbit(params.symm || ((ngrp % 2) != 0), cp);
    grp2.add_orbit(params.symm, pp);

    params.grp2.clear();
    grp2.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_PERM_IMPL_H
