#ifndef LIBTENSOR_SO_MERGE_IMPL_PERM_H
#define LIBTENSOR_SO_MERGE_IMPL_PERM_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../core/permutation_builder.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_merge.h"
#include "../se_perm.h"
#include "permutation_group.h"

namespace libtensor {


/**	\brief Implementation of so_merge<N, M, K, T> for se_perm<N - M + K, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_impl< so_merge<N, M, K, T>, se_perm<N, T> > :
public
symmetry_operation_impl_base< so_merge<N, M, K, T>, se_perm<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_merge<N, M, K, T> operation_t;
	typedef se_perm<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;

};

template<size_t N, size_t M, size_t K, typename T>
const char *
symmetry_operation_impl< so_merge<N, M, K, T>, se_perm<N, T> >::k_clazz =
	"symmetry_operation_impl< so_merge<N, M, K, T>, se_perm<N, T> >";

template<size_t N, size_t M, size_t K, typename T>
void
symmetry_operation_impl< so_merge<N, M, K, T>, se_perm<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

	//	Adapter type for the input group
	//
    typedef se_perm<N - M + K, T> el2_t;
	typedef symmetry_element_set_adapter<N, T, element_t> adapter1_t;

	//	Verify that the projection mask is correct
	//
	size_t nm = 0;
	mask<N> tm, mm; // Total mask and mask of vanishing indexes
	for(size_t k = 0; k < K; k++) {
	    const mask<N> &m = params.msk[k];

	    bool found = false;
	    for (size_t i = 0; i < N; i++) {
	        if (! m[i]) continue;

	        if (tm[i]) {
                throw bad_parameter(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "Masks overlap.");
	        }

	        tm[i] = true;
	        nm++;

	        if (found) mm[i] = true;
	        else found = true;
	    }
	}

	if(nm != M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

	adapter1_t g1(params.grp1);
	permutation_group<N, T> grp1(g1);
	permutation_group<N, T> grp2;
	grp1.stabilize(params.msk, grp2);

    symmetry_element_set<N, T> set(element_t::k_sym_type);
    grp2.convert(set);

    adapter1_t g2(set);
    for (typename adapter1_t::iterator it = g2.begin(); it != g2.end(); it++) {

        const element_t &e2 = g2.get_elem(it);

        sequence<N, size_t> seq1a(0), seq2a(0);
        sequence<N - M + K, size_t> seq1b(0), seq2b(0);

        for (size_t j = 0; j < N; j++) seq1a[j] = seq2a[j] = j;
        e2.get_perm().apply(seq2a);

        for (size_t j = 0, k = 0; j < N; j++) {
            if (mm[j]) continue;

            size_t jj = seq2a[j];
            if (tm[j]) {
                size_t l = 0;
                for (; l < K; l++) {
                    if (params.msk[l][j]) break;
                }
                const mask<N> &m = params.msk[l];
                for (l = j + 1; l < N; l++) {
                    if (! m[l]) continue;

                    jj = std::min(jj, seq2a[l]);
                }
            }

            seq1b[k] = seq1a[j];
            seq2b[k] = jj;
            k++;
        }

        permutation_builder<N - M + K> pb(seq2b, seq1b);
        if (pb.get_perm().is_identity()) {
            if (e2.is_symm()) continue;

            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Anti-symmetric identity permutation.");
        }

        params.grp2.insert(el2_t(pb.get_perm(), e2.is_symm()));
    }
}

/** \brief Implementation of so_merge<N, N, 1, T> for se_perm<1, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_merge<N, N, 1, T>, se_perm<N, T> > :
public symmetry_operation_impl_base< so_merge<N, N, 1, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<N, N, 1, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
        symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

template<size_t N, typename T>
const char *
symmetry_operation_impl< so_merge<N, N, 1, T>, se_perm<N, T> >::k_clazz =
    "symmetry_operation_impl< so_merge<N, N, 1, T>, se_perm<N, T> >";

template<size_t N, typename T>
void symmetry_operation_impl< so_merge<N, N, 1, T>, se_perm<N, T> >
::do_perform(symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

	//	Verify that the projection mask is correct
	//
	const mask<N> &m = params.msk[0];
	size_t nm = 0;
	for(size_t i = 0; i < N; i++) if(m[i]) nm++;

	if(nm != N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

    //  Adapter type for the input group
    //
    typedef se_perm<1, T> el2_t;
    typedef symmetry_element_set_adapter<N, T, element_t> adapter1_t;

    adapter1_t g1(params.grp1);
    for (typename adapter1_t::iterator it = g1.begin(); it != g1.end(); it++) {

        const element_t &e1 = g1.get_elem(it);
        if (! e1.is_symm()) {
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Anti-symmetric identity permutation.");
//            params.grp2.insert(el2_t(permutation<1>(), e1.is_symm()));
//            break;
        }
    }

}

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_PERM_H
