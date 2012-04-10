#ifndef LIBTENSOR_SO_DIRSUM_SE_PART_IMPL_H
#define LIBTENSOR_SO_DIRSUM_SE_PART_IMPL_H

#include <map>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/permutation_builder.h>
#include "../combine_part.h"

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_dirsum<N, M, T>, se_part<N + M, T> >::k_clazz =
        "symmetry_operation_impl< so_dirsum<N, M, T>, se_part<N + M, T> >";

template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_dirsum<N, M, T>, se_part<N + M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    params.g3.clear();

    // Identity transform
    scalar_transf<T> tr0;

    // Nothing needs to be done if both sets are empty
    if (params.g1.is_empty() && params.g2.is_empty()) return;

    // [120] map2[

    permutation<N + M> pinv(params.perm, true);
    sequence<N, size_t> map1(0);
    sequence<M, size_t> map2(0);
    for (register size_t i = 0; i < N; i++) map1[i] = pinv[i];
    for (register size_t i = 0; i < M; i++) map2[i] = pinv[i + N];

    // First set is empty
    if (params.g1.is_empty()) {
        combine_part<M, T> p2(params.g2);
        se_part<M, T> se2(p2.get_bis(), p2.get_pdims());
        p2.perform(se2);

        index<N + M> i3a, i3b;
        const dimensions<M> &pdims2 = se2.get_pdims();
        for (size_t i = 0; i < M; i++) i3b[map2[i]] = pdims2[i] - 1;

        se_part<N + M, T> se3(params.bis,
                dimensions<N + M>(index_range<N + M>(i3a, i3b)));

        abs_index<M> ai2a(pdims2);
        do {
            const index<M> &i2a = ai2a.get_index();
            if (se2.is_forbidden(i2a)) {
                abs_index<M> ai2b(i2a, pdims2);
                while (ai2b.inc() && ! se2.is_forbidden(ai2b.get_index()));

                const index<M> &i2b = ai2b.get_index();
                if (! se2.is_forbidden(i2b)) continue;

                for (size_t i = 0; i < M; i++) {
                    i3a[map2[i]] = i2a[i];
                    i3b[map2[i]] = i2b[i];
                }
                se3.add_map(i3a, i3b, tr0);
                continue;
            }

            index<M> i2b = se2.get_direct_map(i2a);
            while (i2a < i2b) {
                scalar_transf<T> tr = se2.get_transf(i2a, i2b);
                if (tr.is_identity()) {
                    for (register size_t i = 0; i < M; i++) {
                        i3a[map2[i]] = i2a[i];
                        i3b[map2[i]] = i2b[i];
                    }
                    se3.add_map(i3a, i3b, tr);
                    break;
                }
                i2b = se2.get_direct_map(i2b);
            }

        } while (ai2a.inc());

        params.g3.insert(se3);
    }
    else if (params.g2.is_empty()) {

        combine_part<N, T> p1(params.g1);
        se_part<N, T> se1(p1.get_bis(), p1.get_pdims());
        p1.perform(se1);

        index<N + M> i3a, i3b;
        const dimensions<N> &pdims1 = se1.get_pdims();
        for (size_t i = 0; i < N; i++) { i3b[map1[i]] = pdims1[i] - 1; }

        se_part<N + M, T> se3(params.bis,
                dimensions<N + M>(index_range<N + M>(i3a, i3b)));

        abs_index<N> ai1a(pdims1);
        do {
            const index<N> &i1a = ai1a.get_index();
            if (se1.is_forbidden(i1a)) {
                abs_index<N> ai1b(i1a, pdims1);
                while (ai1b.inc() && ! se1.is_forbidden(ai1b.get_index()));

                const index<N> &i1b = ai1b.get_index();
                if (! se1.is_forbidden(i1b)) continue;

                for (size_t i = 0; i < N; i++) {
                    i3a[map1[i]] = i1a[i];
                    i3b[map1[i]] = i1b[i];
                }
                se3.add_map(i3a, i3b, tr0);
                continue;
            }

            index<N> i1b = se1.get_direct_map(i1a);
            while (i1a < i1b) {
                scalar_transf<T> tr = se1.get_transf(i1a, i1b);
                if (tr.is_identity()) {
                    for (register size_t i = 0; i < N; i++) {
                        i3a[map1[i]] = i1a[i];
                        i3b[map1[i]] = i1b[i];
                    }
                    se3.add_map(i3a, i3b, tr);
                    break;
                }
                i1b = se1.get_direct_map(i1b);
            }
        } while (ai1a.inc());

        params.g3.insert(se3);
    }
    else {
        // Merge symmetry element set 1 into one se_part
        combine_part<N, T> p1(params.g1);
        se_part<N, T> se1(p1.get_bis(), p1.get_pdims());
        p1.perform(se1);

        // Merge symmetry element set 2 into one se_part
        combine_part<M, T> p2(params.g2);
        se_part<M, T> se2(p2.get_bis(), p2.get_pdims());
        p2.perform(se2);

        // Build the partition dimensions of the result
        index<N + M> i3a, i3b;

        const dimensions<N> &pdims1 = se1.get_pdims();
        for (size_t i = 0; i < N; i++) i3b[map1[i]] = pdims1[i] - 1;
        const dimensions<M> &pdims2 = se2.get_pdims();
        for (size_t i = 0; i < M; i++) i3b[map2[i]] = pdims2[i] - 1;

        // Construct the result
        se_part<N + M, T> se3(params.bis,
                dimensions<N + M>(index_range<N + M>(i3a, i3b)));

        abs_index<N + M> ai3a(se3.get_pdims());
        do {
            const index<N + M> &i3a = ai3a.get_index();

            index<N> i1a;
            for (register size_t i = 0; i < N; i++) i1a[i] = i3a[map1[i]];
            index<M> i2a;
            for (register size_t i = 0; i < M; i++) i2a[i] = i3a[map2[i]];

            bool fb1 = se1.is_forbidden(i1a);
            bool fb2 = se2.is_forbidden(i2a);

            if (fb1 && fb2) { // Both forbidden
                se3.mark_forbidden(i3a);
                continue;
            }

            if (fb1) { // se1 forbidden
                // Try to create a map leaving se1 indexes fixed
                index<M> i2b = se2.get_direct_map(i2a);
                if (i2a < i2b) {
                    for (register size_t i = 0; i < N; i++)
                        i3b[map1[i]] = i1a[i];
                    for (register size_t i = 0; i < M; i++)
                        i3b[map2[i]] = i2b[i];
                    se3.add_map(i3a, i3b, se2.get_transf(i2a, i2b));

                    continue;
                }

                // Otherwise check, if there is another forbidden index in se1
                abs_index<N> ai1b(i1a, se1.get_pdims());
                while (ai1b.inc()) {
                    if (se1.is_forbidden(ai1b.get_index())) break;
                }

                if (se1.is_forbidden(ai1b.get_index())) {
                    const index<N> &i1b = ai1b.get_index();
                    for (register size_t i = 0; i < N; i++)
                        i3b[map1[i]] = i1b[i];
                    for (register size_t i = 0; i < M; i++)
                        i3b[map2[i]] = i2a[i];
                    se3.add_map(i3a, i3b, tr0);
                }

                continue;
            }

            if (fb2) { // se2 forbidden
                // Check, if there is another forbidden index in se2
                abs_index<M> ai2b(i2a, se2.get_pdims());
                while (ai2b.inc()) {
                    if (se2.is_forbidden(ai2b.get_index())) break;
                }

                // If there is, create a map
                if (se2.is_forbidden(ai2b.get_index())) {
                    const index<M> &i2b = ai2b.get_index();
                    for (register size_t i = 0; i < N; i++)
                        i3b[map1[i]] = i1a[i];
                    for (register size_t i = 0; i < M; i++)
                        i3b[map2[i]] = i2b[i];
                    se3.add_map(i3a, i3b, tr0);

                    continue;
                }

                // Try to create a map leaving se2 indexes fixed
                index<N> i1b = se1.get_direct_map(i1a);
                if (i1a < i1b) {
                    for (register size_t i = 0; i < N; i++)
                        i3b[map1[i]] = i1b[i];
                    for (register size_t i = 0; i < M; i++)
                        i3b[map2[i]] = i2a[i];
                    se3.add_map(i3a, i3b, se1.get_transf(i1a, i1b));
                }

                continue;
            }

            // Both allowed
            // First try to create a map which leaves indexes of 1 fixed
            index<M> i2b = se2.get_direct_map(i2a);
            scalar_transf<T> tr2 = se2.get_transf(i2a, i2b);
            while (i2a < i2b && ! tr2.is_identity()) {
                i2b = se2.get_direct_map(i2b);
                tr2 = se2.get_transf(i2a, i2b);
            }
            if (i2a < i2b) {
                for (register size_t i = 0; i < N; i++)
                    i3b[map1[i]] = i1a[i];
                for (register size_t i = 0; i < M; i++)
                    i3b[map2[i]] = i2b[i];
                se3.add_map(i3a, i3b, tr2);

                continue;
            }

            // Then try to find the map that changes the indexes of 1 by the
            // least amount
            index<N> i1b = se1.get_direct_map(i1a);
            while (i1a < i1b) {
                scalar_transf<T> tr1 = se1.get_transf(i1a, i1b);
                // This can either be changeing only indexes of 1
                if (tr1.is_identity()) {
                    for (register size_t i = 0; i < N; i++)
                        i3b[map1[i]] = i1b[i];
                    for (register size_t i = 0; i < M; i++)
                        i3b[map2[i]] = i2a[i];
                    se3.add_map(i3a, i3b, tr1);

                    break;
                }

                // Or of 1 and 2
                index<M> i2b = se2.get_direct_map(i2a);
                while (i2a != i2b) {
                    if (se2.get_transf(i2a, i2b) == tr1) break;
                    i2b = se2.get_direct_map(i2b);
                }
                if (i2a != i2b) {
                    for (register size_t i = 0; i < N; i++)
                        i3b[map1[i]] = i1b[i];
                    for (register size_t i = 0; i < M; i++)
                        i3b[map2[i]] = i2b[i];
                    se3.add_map(i3a, i3b, tr1);

                    break;
                }

                i1b = se1.get_direct_map(i1b);
            }
         } while (ai3a.inc());

        params.g3.insert(se3);
    }
}




} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_SE_PART_IMPL_H
