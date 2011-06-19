#ifndef LIBTENSOR_SO_DIRPROD_IMPL_PERM_H
#define LIBTENSOR_SO_DIRPROD_IMPL_PERM_H

#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_dirprod.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_dirprod<N, M, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	Constructs the direct product of two permutation %symmetry groups.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl<
so_dirprod<N, M, T>, se_perm<N + M, T> > :
public symmetry_operation_impl_base<
so_dirprod<N, M, T>, se_perm<N + M, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_dirprod<N, M, T> operation_t;
    typedef se_perm<N + M, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl<
so_dirprod<N, M, T>, se_perm<N + M, T> >::k_clazz =
        "symmetry_operation_impl< so_dirprod<N, M, T>, se_perm<N, T> >";

template<size_t N, size_t M, typename T>
void symmetry_operation_impl<
so_dirprod<N, M, T>, se_perm<N + M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    //	Adapter type for the input group
    typedef symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter1_t;
    typedef symmetry_element_set_adapter< M, T, se_perm<M, T> > adapter2_t;

    sequence<N + M, size_t> map(0);
    for (size_t j = 0; j < N + M; j++) map[j] = j;
    permutation<N + M> pinv(params.perm, true);
    pinv.apply(map);

    adapter1_t g1(params.g1);

    //	Go over each element in the first source group and project up
    for(typename adapter1_t::iterator i = g1.begin(); i != g1.end(); i++) {

        const se_perm<N, T> &e1 = g1.get_elem(i);

        //	Project the combined permutation onto the larger
        //	space and form a symmetry element
        sequence<N, size_t> a1(0);
        sequence<N + M, size_t> a2a(0), a2b(0);
        for (size_t j = 0; j < N; j++) a1[j] = j;
        e1.get_perm().apply(a1);

        size_t k = 0;
        for(; k < N; k++) {
            a2a[map[k]] = k; a2b[map[k]] = a1[k];
        }
        for(; k < N + M; k++) {
            a2a[map[k]] = a2b[map[k]] = k;
        }

        permutation_builder<N + M> pb(a2b, a2a);
        se_perm<N + M, T> e3(pb.get_perm(), e1.is_symm());
        params.g3.insert(e3);
    }

    adapter2_t g2(params.g2);

    //	Do the same for the second source group
    for(typename adapter2_t::iterator i = g2.begin(); i != g2.end(); i++) {

        const se_perm<M, T> &e2 = g2.get_elem(i);

        //	Project the combined permutation onto the larger
        //	space and form a symmetry element
        sequence<M, size_t> a1(0);
        sequence<N + M, size_t> a2a(0), a2b(0);
        for (size_t j = 0; j < M; j++) a1[j] = N + j;
        e2.get_perm().apply(a1);

        size_t k = 0;
        for(; k < N; k++) {
            a2a[map[k]] = a2b[map[k]] = k;
        }
        for(; k < N + M; k++) {
            a2a[map[k]] = k; a2b[map[k]] = a1[k - N];
        }

        permutation_builder<N + M> pb(a2b, a2a);
        element_t e3(pb.get_perm(), e2.is_symm());
        params.g3.insert(e3);
    }

}


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_IMPL_PERM_H
