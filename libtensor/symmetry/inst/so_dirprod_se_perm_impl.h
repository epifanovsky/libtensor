#ifndef LIBTENSOR_SO_DIRPROD_SE_PERM_IMPL_H
#define LIBTENSOR_SO_DIRPROD_SE_PERM_IMPL_H

#include <libtensor/core/permutation_builder.h>

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_dirprod<N, M, T>, se_perm<N + M, T> >::k_clazz =
        "symmetry_operation_impl< so_dirprod<N, M, T>, se_perm<N + M, T> >";

template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_dirprod<N, M, T>, se_perm<N + M, T> >::do_perform(
    symmetry_operation_params_t &params) const {

    //  Adapter type for the input group
    typedef symmetry_element_set_adapter<N, T, se_perm<N, T> > adapter1_t;
    typedef symmetry_element_set_adapter<M, T, se_perm<M, T> > adapter2_t;

    params.g3.clear();

    sequence<N + M, size_t> map(0);
    for (size_t j = 0; j < N + M; j++) map[j] = j;
    permutation<N + M> pinv(params.perm, true);
    pinv.apply(map);

    adapter1_t g1(params.g1);

    //  Go over each element in the first source group and project up
    for(typename adapter1_t::iterator i = g1.begin(); i != g1.end(); i++) {

        const se_perm<N, T> &e1 = g1.get_elem(i);

        //  Project the permutation onto the larger
        //  space and form a symmetry element
        sequence<N + M, size_t> a2a(0), a2b(0);

        for(register size_t k = 0; k < N; k++) {
            a2a[map[k]] = k; a2b[map[k]] = e1.get_perm()[k];
        }
        for(register size_t k = N; k < N + M; k++)
            a2a[map[k]] = a2b[map[k]] = k;

        permutation_builder<N + M> pb(a2b, a2a);
        se_perm<N + M, T> e3(pb.get_perm(), e1.get_transf());
        params.g3.insert(e3);
    }

    adapter2_t g2(params.g2);

    //  Do the same for the second source group
    for(typename adapter2_t::iterator i = g2.begin(); i != g2.end(); i++) {

        const se_perm<M, T> &e2 = g2.get_elem(i);

        //  Project the permutation onto the larger
        //  space and form a symmetry element
        sequence<N + M, size_t> a2a(0), a2b(0);
        for(register size_t k = 0; k < N; k++) a2a[map[k]] = a2b[map[k]] = k;

        for(register size_t k = N, l = 0; k < N + M; k++, l++) {
            a2a[map[k]] = k; a2b[map[k]] = e2.get_perm()[l] + N;
        }

        permutation_builder<N + M> pb(a2b, a2a);
        element_t e3(pb.get_perm(), e2.get_transf());
        params.g3.insert(e3);
    }

}


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_SE_PERM_IMPL_H
