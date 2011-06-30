#ifndef LIBTENSOR_SO_DIRSUM_IMPL_PERM_H
#define LIBTENSOR_SO_DIRSUM_IMPL_PERM_H

#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_dirsum.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_dirsum<N, M, T> for se_perm<N + M, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N + M, T> > :
public symmetry_operation_impl_base< so_dirsum<N, M, T>, se_perm<N + M, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_dirsum<N, M, T> operation_t;
    typedef se_perm<N + M, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl< 
so_dirsum<N, M, T>, se_perm<N + M, T> >::k_clazz =
        "symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< 
so_dirsum<N, M, T>, se_perm<N + M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    //	Adapter type for the input group
    typedef symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter1_t;
    typedef symmetry_element_set_adapter< M, T, se_perm<M, T> > adapter2_t;

    sequence<N + M, size_t> map(0);
    for (size_t k = 0; k < N + M; k++) map[k] = k;
    permutation<N + M> pinv(params.perm, true);
    pinv.apply(map);

    permutation_group<N + M, double> group;

    adapter1_t g1(params.g1);
    adapter2_t g2(params.g2);

    // Loop over all elements in the first source group
    for(typename adapter1_t::iterator i = g1.begin(); i != g1.end(); i++) {

        const se_perm<N, T> &e1 = g1.get_elem(i);
        // Project up symmetric permutations
        if (e1.is_symm()) {

            sequence<N, size_t> a1(0);
            sequence<N + M, size_t> a2a(0), a2b(0);
            for (size_t k = 0; k < N; k++) a1[k] = k;
            e1.get_perm().apply(a1);

            for(size_t k = 0; k < N; k++) {
                a2a[map[k]] = k; a2b[map[k]] = a1[k];
            }
            for(size_t k = N; k < N + M; k++) {
                a2a[map[k]] = a2b[map[k]] = k;
            }

            permutation_builder<N + M> pb(a2b, a2a);
            group.add_orbit(true, pb.get_perm());
        }
        else {
            // Try to combine two anti-symmetric permutations and project up
            for (typename adapter1_t::iterator j = i; j != g1.end(); j++) {

                const se_perm<N, T> &e1b = g1.get_elem(j);
                if (e1b.is_symm()) continue;

                permutation<N> p(e1.get_perm());
                p.permute(e1b.get_perm());
                if (! p.is_identity()) continue;

                sequence<N, size_t> a1(0);
                sequence<N + M, size_t> a2a(0), a2b(0);
                for (size_t k = 0; k < N; k++) a1[k] = k;
                p.apply(a1);

                for(size_t k = 0; k < N; k++) {
                    a2a[map[k]] = k; a2b[map[k]] = a1[k];
                }
                for(size_t k = N; k < N + M; k++) {
                    a2a[map[k]] = a2b[map[k]] = k;
                }

                permutation_builder<N + M> pb(a2b, a2a);
                group.add_orbit(true, pb.get_perm());
            }
        }
    }


    //	Do the same for the second source group
    for(typename adapter2_t::iterator i = g2.begin(); i != g2.end(); i++) {

        const se_perm<M, T> &e2 = g2.get_elem(i);
        if (e2.is_symm()) {

            sequence<M, size_t> a1(0);
            sequence<N + M, size_t> a2a(0), a2b(0);
            for (size_t k = 0; k < M; k++) a1[k] = N + k;
            e2.get_perm().apply(a1);

            for(size_t k = 0; k < N; k++) {
                a2a[map[k]] = a2b[map[k]] = k;
            }
            for(size_t k = N; k < N + M; k++) {
                a2a[map[k]] = k; a2b[map[k]] = a1[k - N];
            }

            permutation_builder<N + M> pb(a2b, a2a);
            group.add_orbit(true, pb.get_perm());
        }
        else {
            for (typename adapter2_t::iterator j = i; j != g2.end(); j++) {

                const se_perm<M, T> &e2b = g2.get_elem(j);
                if (e2b.is_symm()) continue;

                permutation<M> p(e2.get_perm());
                p.permute(e2b.get_perm());
                if (p.is_identity()) continue;

                sequence<M, size_t> a1(0);
                sequence<N + M, size_t> a2a(0), a2b(0);
                for (size_t k = 0; k < M; k++) a1[k] = N + k;
                p.apply(a1);

                for(size_t k = 0; k < N; k++) {
                    a2a[map[k]] = a2b[map[k]] = k;
                }
                for(size_t k = N; k < N + M; k++) {
                    a2a[map[k]] = k; a2b[map[k]] = a1[k - N];
                }

                permutation_builder<N + M> pb(a2b, a2a);
                group.add_orbit(true, pb.get_perm());

            }
        }
    }

    // Now loop over all anti-symmetric permutations in both source groups
    // and create composite permutations
    for(typename adapter1_t::iterator i = g1.begin(); i != g1.end(); i++) {

        const se_perm<N, T> &e1 = g1.get_elem(i);
        if (e1.is_symm()) continue;

        sequence<N, size_t> a1(0);
        for (size_t k = 0; k < N; k++) a1[k] = k;
        e1.get_perm().apply(a1);

        for(typename adapter2_t::iterator j = g2.begin(); j != g2.end(); j++) {

            const se_perm<M, T> &e2 = g2.get_elem(j);
            if (e2.is_symm()) continue;

            sequence<M, size_t> a2(0);
            for (size_t k = 0; k < M; k++) a2[k] = N + k;
            e2.get_perm().apply(a2);

            sequence<N + M, size_t> a3a(0), a3b(0);
            for(size_t k = 0; k < N; k++) {
                a3a[map[k]] = k; a3b[map[k]] = a1[k];
            }
            for(size_t k = N; k < N + M; k++) {
                a3a[map[k]] = k; a3b[map[k]] = a2[k - N];
            }

            permutation_builder<N + M> pb(a3b, a3a);
            group.add_orbit(false, pb.get_perm());
        }
    }

    group.convert(params.g3);
}

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_IMPL_PERM_H
