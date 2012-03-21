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

    // Nothing needs to be done if both sets are empty
    if (params.g1.is_empty() && params.g2.is_empty()) return;

    // Map where which input index ends up in the output
    sequence<N + M, size_t> map(0);
    for (size_t i = 0; i < N + M; i++) map[i] = i;
    permutation<N + M> pinv(params.perm, true);
    pinv.apply(map);

    if (params.g1.is_empty()) {

        combine_part<M, T> p2(params.g2);
        se_part<M, T> se2(p2.get_bis(), p2.get_pdims());
        p2.perform(se2);

        index<N + M> i3a, i3b;
        const dimensions<M> &pdims2 = se2.get_pdims();
        for (size_t i = 0; i < M; i++) { i3b[map[i + N]] = pdims2[i] - 1; }

        se_part<N + M, T> se3(params.bis,
                dimensions<N + M>(index_range<N + M>(i3a, i3b)));

        std::list< index<M> > flist2;

        abs_index<M> ai2(pdims2);
        do {
            const index<M> &i2a = ai2.get_index();
            if (se2.is_forbidden(i2a)) {
                flist2.push_back(i2a);
                continue;
            }

            index<M> i2b = se2.get_direct_map(i2a);
            if (! (i2a < i2b)) continue;

            bool sign = se2.get_sign(i2a, i2b);
            if (! sign) continue;

            for (register size_t i = 0; i < M; i++) {
                i3a[map[i + N]] = i2a[i];
                i3b[map[i + N]] = i2b[i];
            }
            se3.add_map(i3a, i3b, sign);

        } while (ai2.inc());

        if (flist2.size() > 1) {

            typename std::list< index<M> >::iterator it1 = flist2.begin();
            typename std::list< index<M> >::iterator it2 = flist2.begin();
            it2++;
            for (; it2 != flist2.end(); it1++, it2++) {

                for (register size_t i = 0; i < M; i++) {
                    i3a[map[i + N]] = (*it1)[i];
                    i3b[map[i + N]] = (*it2)[i];
                }
                se3.add_map(i3a, i3b, true);
            }
        }

        params.g3.insert(se3);
    }
    else if (params.g2.is_empty()) {

        combine_part<N, T> p1(params.g1);
        se_part<N, T> se1(p1.get_bis(), p1.get_pdims());
        p1.perform(se1);

        index<N + M> i3a, i3b;
        const dimensions<N> &pdims1 = se1.get_pdims();
        for (size_t i = 0; i < N; i++) { i3b[map[i]] = pdims1[i] - 1; }

        se_part<N + M, T> se3(params.bis,
                dimensions<N + M>(index_range<N + M>(i3a, i3b)));

        std::list< index<N> > flist1;

        abs_index<N> ai1(pdims1);
        do {
            const index<N> &i1a = ai1.get_index();
            if (se1.is_forbidden(i1a)) {
                flist1.push_back(i1a);
                continue;
            }

            index<N> i1b = se1.get_direct_map(i1a);
            if (! (i1a < i1b)) continue;

            if (! se1.get_sign(i1a, i1b)) continue;

            for (register size_t i = 0; i < N; i++) {
                i3a[map[i]] = i1a[i];
                i3b[map[i]] = i1b[i];
            }
            se3.add_map(i3a, i3b, true);

        } while (ai1.inc());

        if (flist1.size() > 1) {

            typename std::list< index<N> >::iterator it1 = flist1.begin();
            typename std::list< index<N> >::iterator it2 = flist1.begin();
            it2++;
            for (; it2 != flist1.end(); it1++, it2++) {

                for (register size_t i = 0; i < N; i++) {
                    i3a[map[i]] = (*it1)[i];
                    i3b[map[i]] = (*it2)[i];
                }
                se3.add_map(i3a, i3b, true);
            }
        }

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
        for (size_t i = 0; i < N; i++) i3b[map[i]] = pdims1[i] - 1;
        const dimensions<M> &pdims2 = se2.get_pdims();
        for (size_t i = 0; i < M; i++) i3b[map[i + N]] = pdims2[i] - 1;

        // Construct the result
        se_part<N + M, T> se3(params.bis,
                dimensions<N + M>(index_range<N + M>(i3a, i3b)));


        abs_index<N + M> ai3a(se3.get_pdims());
        do {
            const index<N + M> &i3a = ai3a.get_index();

            index<N> i1a, i1b;
            for (register size_t i = 0; i < N; i++) {
                i1a[i] = i3a[map[i]];
            }

            index<M> i2a, i2b;
            for (register size_t i = 0; i < M; i++) {
                i2a[i] = i3a[map[i + N]];
            }

            bool fb1 = se1.is_forbidden(i1a);
            bool fb2 = se2.is_forbidden(i2a);

            bool s1, s2;

            if (fb1 && fb2) { // Both forbidden
                se3.mark_forbidden(i3a);
                continue;
            }
            else if (fb1) { // se1 forbidden

                // Check, if there is another forbidden index in se1
                abs_index<N> ai1b(i1a, se1.get_pdims());
                while (ai1b.inc()) {
                    if (se1.is_forbidden(ai1b.get_index())) break;
                }

                // If yes, create a map
                if (se1.is_forbidden(ai1b.get_index())) {
                    i1b = ai1b.get_index();
                    s1 = true;
                }

                // Create a map leaving se1 indexes fixed
                i2b = se2.get_direct_map(i2a);
                s2 = se2.get_sign(i2a, i2b);

            }
            else if (fb2) { // se2 forbidden

                // Check, if there is another forbidden index in se2
                abs_index<M> ai2b(i2a, se2.get_pdims());
                while (ai2b.inc()) {
                    if (se2.is_forbidden(ai2b.get_index())) break;
                }

                // If there is, create a map
                if (se2.is_forbidden(ai2b.get_index())) {
                    i2b = ai2b.get_index();
                    s2 = true;
                }

                // Create a map leaving se1 indexes fixed
                i1b = se1.get_direct_map(i1a);
                s1 = se1.get_sign(i1a, i1b);
            }
            else { // Both allowed

                i1b = se1.get_direct_map(i1a);
                s1 = se1.get_sign(i1a, i1b);

                i2b = se2.get_direct_map(i2a);
                s2 = se2.get_sign(i2a, i2b);
            }

            if ((i1a < i1b) && (s1 || fb2)) {

                for (register size_t i = 0; i < N; i++) {
                    i3b[map[i]] = i1b[i];
                }
                for (register size_t i = 0; i < M; i++) {
                    i3b[map[i + N]] = i3a[map[i + N]];
                }

                se3.add_map(i3a, i3b, s1);
            }

            if ((i2a < i2b) && (s2 || fb1)) {

                for (register size_t i = 0; i < N; i++) {
                    i3b[map[i]] = i3a[map[i]];
                }
                for (register size_t i = 0; i < M; i++) {
                    i3b[map[i + N]] = i2b[i];
                }

                se3.add_map(i3a, i3b, s2);
            }

            if (!fb1 && !fb2 && s1 == s2 && !s1) {

                if (! (i1a < i1b && i2a < i2b)) continue;

                for (register size_t i = 0; i < N; i++) {
                    i3b[map[i]] = i1b[i];
                }
                for (register size_t i = 0; i < M; i++) {
                    i3b[map[i + N]] = i2b[i];
                }

                se3.add_map(i3a, i3b, s1);

                // One last case to add:
                index<N + M> i3a2;
                for (register size_t i = 0; i < N; i++) {
                    i3a2[map[i]] = i1a[i]; i3b[map[i]] = i1b[i];
                }
                for (register size_t i = 0; i < M; i++) {
                    i3a2[map[i + N]] = i2b[i]; i3b[map[i + N]] = i2a[i];
                }

                se3.add_map(i3a2, i3b, s1);
            }
        } while (ai3a.inc());

        params.g3.insert(se3);
    }

}

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_SE_PART_IMPL_H
