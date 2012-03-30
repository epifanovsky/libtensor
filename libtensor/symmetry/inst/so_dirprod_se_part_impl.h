#ifndef LIBTENSOR_SO_DIRPROD_SE_PART_IMPL_H
#define LIBTENSOR_SO_DIRPROD_SE_PART_IMPL_H

#include <map>
#include <libtensor/defs.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/permutation_builder.h>

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_dirprod<N, M, T>, se_part<N + M, T> >::k_clazz =
        "symmetry_operation_impl< so_dirprod<N, M, T>, se_part<N, T> >";


template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_dirprod<N, M, T>, se_part<N + M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    // Adapter type for the input groups
    typedef symmetry_element_set_adapter< N, T, se_part<N, T> > adapter1_t;
    typedef symmetry_element_set_adapter< M, T, se_part<M, T> > adapter2_t;

    adapter1_t g1(params.g1);
    adapter2_t g2(params.g2);
    params.g3.clear();

    // Create map that tells the position where which index ends up.
    sequence<N + M, size_t> map(0);
    for (size_t i = 0; i < N + M; i++) map[i] = i;
    permutation<N + M> pinv(params.perm, true);
    pinv.apply(map);

    index<N + M> tot_pdims;
    for (size_t i = 0; i < N + M; i++) tot_pdims[i] = 1;

    for(typename adapter1_t::iterator it1 = g1.begin();
            it1 != g1.end(); it1++) {

        const se_part<N, T> &e1 = g1.get_elem(it1);
        const dimensions<N> &pdims1 = e1.get_pdims();

        index<N + M> i3a, i3b;
        for (size_t i = 0; i < N; i++) {

#ifdef LIBTENSOR_DEBUG
            if (pdims1[i] != 1) {
                if ((tot_pdims[map[i]] != 1) &&
                        (tot_pdims[map[i]] != pdims1[i])) {
                    throw bad_symmetry(g_ns, k_clazz, method, __FILE__,
                            __LINE__, "Illegal pdims in g1.");
                }
                tot_pdims[map[i]] = pdims1[i];
            }
#endif
            i3b[map[i]] = pdims1[i] - 1;
        }

        se_part<N + M, T> e3(params.bis,
                dimensions<N + M>(index_range<N + M>(i3a, i3b)));

        abs_index<N> ai(pdims1);
        do {
            const index<N> &i1a = ai.get_index();

            for (size_t i = 0; i < N; i++) i3a[map[i]] = i1a[i];

            if (e1.is_forbidden(i1a)) {
                e3.mark_forbidden(i3a);
                continue;
            }

            index<N> i1b = e1.get_direct_map(i1a);
            if (i1a == i1b) continue;

            for (size_t i = 0; i < N; i++) i3b[map[i]] = i1b[i];

            e3.add_map(i3a, i3b, e1.get_transf(i1a, i1b));

        } while (ai.inc());

        params.g3.insert(e3);
    }

    for(typename adapter2_t::iterator it2 = g2.begin();
            it2 != g2.end(); it2++) {

        const se_part<M, T> &e2 = g2.get_elem(it2);
        const dimensions<M> &pdims2 = e2.get_pdims();

        index<N + M> i3a, i3b;
        for (size_t i = 0; i < M; i++) {
#ifdef LIBTENSOR_DEBUG
            if (pdims2[i] != 1) {
                if ((tot_pdims[map[i + N]] != 1) &&
                        (tot_pdims[map[i + N]] != pdims2[i])) {
                    throw bad_symmetry(g_ns, k_clazz, method, __FILE__,
                            __LINE__, "Illegal pdims in g2.");
                }
                tot_pdims[map[i + N]] = pdims2[i];
            }
#endif
            i3b[map[i + N]] = pdims2[i] - 1;
        }

        se_part<N + M, T> e3(params.bis,
                dimensions<N + M>(index_range<N + M>(i3a, i3b)));

        abs_index<M> ai(e2.get_pdims());
        do {
            const index<M> &i2a = ai.get_index();

            for (size_t i = 0; i < M; i++) i3a[map[i + N]] = i2a[i];

            if (e2.is_forbidden(i2a)) {
                e3.mark_forbidden(i3a);
                continue;
            }

            index<M> i2b = e2.get_direct_map(i2a);
            if (i2a == i2b) continue;

            index<N + M> i3b;
            for (size_t i = 0; i < M; i++) i3b[map[i + N]] = i2b[i];

            e3.add_map(i3a, i3b, e2.get_transf(i2a, i2b));

        } while (ai.inc());

        params.g3.insert(e3);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_SE_PART_IMPL_H
