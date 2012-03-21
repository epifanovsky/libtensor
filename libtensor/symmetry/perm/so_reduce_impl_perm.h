#ifndef LIBTENSOR_SO_REDUCE_IMPL_PERM_H
#define LIBTENSOR_SO_REDUCE_IMPL_PERM_H

#include "../../core/permutation_builder.h"
#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_reduce.h"
#include "../se_perm.h"
#include "permutation_group.h"

namespace libtensor {


/**	\brief Implementation of so_reduce<N, M, K, T> for se_perm<N, T>
	\tparam N Input tensor order.
	\tparam M Reduced dimensions.
    \tparam K Reduction steps.
	\tparam T Tensor element type.

	The implementation reduces masked dimensions setwise.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_impl< so_reduce<N, M, K, T>, se_perm<N, T> > :
public symmetry_operation_impl_base< so_reduce<N, M, K, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_reduce<N, M, K, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

template<size_t N, size_t M, size_t K, typename T>
const char *symmetry_operation_impl< so_reduce<N, M, K, T>, se_perm<N, T> >
::k_clazz = "symmetry_operation_impl< so_reduce<N, M, K, T>, se_perm<N, T> >";

template<size_t N, size_t M, size_t K, typename T>
void symmetry_operation_impl< so_reduce<N, M, K, T>, se_perm<N, T> >
::do_perform(symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(symmetry_operation_params_t&)";

    //	Adapter type for the input group
    //
    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;
    typedef se_perm<N - M, T> el2_t;

    //	Verify that the projection mask is correct
    //
    mask<N> tm;
    size_t m = 0;
    for (size_t k = 0; k < K; k++) {
        const mask<N> &msk = params.msk[k];
        for(size_t i = 0; i < N; i++) {
            if(!msk[i]) continue;
            if(tm[i])
                throw bad_parameter(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "params.msk[k]");

            tm[i] = true;
            m++;
        }
    }
    if(m != M)
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "params.msk");

    adapter_t adapter1(params.grp1);
    permutation_group<N, T> grp1(adapter1);
    permutation_group<N, T> grp2;
    grp1.stabilize(params.msk, grp2);

    symmetry_element_set<N, T> set(element_t::k_sym_type);
    grp2.convert(set);

    adapter_t g2(set);
    params.grp2.clear();
    for (typename adapter_t::iterator it = g2.begin(); it != g2.end(); it++) {
        const element_t &e2 = g2.get_elem(it);

        sequence<N, size_t> seq1a(0), seq2a(0);
        sequence<N - M, size_t> seq1b(0), seq2b(0);

        for (size_t j = 0; j < N; j++) seq1a[j] = seq2a[j] = j;
        e2.get_perm().apply(seq2a);

        for (size_t j = 0, k = 0; j < N; j++) {
            if (tm[j]) continue;

            seq1b[k] = seq1a[j];
            seq2b[k] = seq2a[j];
            k++;
        }

        permutation_builder<N - M> pb(seq2b, seq1b);
        if (pb.get_perm().is_identity()) {
            if (e2.is_symm()) continue;

            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Anti-symmetric identity permutation.");
        }

        params.grp2.insert(el2_t(pb.get_perm(), e2.is_symm()));
    }
}



} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_IMPL_PERM_H
