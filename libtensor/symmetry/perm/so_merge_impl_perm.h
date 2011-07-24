#ifndef LIBTENSOR_SO_MERGE_IMPL_PERM_H
#define LIBTENSOR_SO_MERGE_IMPL_PERM_H

#include <list>
#include "../../defs.h"
#include "../../exception.h"
#include "../../core/permutation_builder.h"
#include "../se_perm.h"
#include "../so_merge.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "permutation_group.h"

namespace libtensor {


/**	\brief Implementation of so_merge<N, M, K, T> for se_perm<N - M + K, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_impl< so_merge<N, M, K, T>, se_perm<N - M + K, T> > :
	public symmetry_operation_impl_base<
		so_merge<N, M, K, T>, se_perm<N - M + K, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_merge<N, M, K, T> operation_t;
	typedef se_perm<N - M + K, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;

};

/** \brief Implementation of so_merge<2 N, 2 N, N, T> for se_perm<N, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_merge<2 * N, 2 * N, N, T>, se_perm<N, T> > :
    public symmetry_operation_impl_base<
        so_merge<2 * N, 2 * N, N, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<2 * N, 2 * N, N, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
        symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

/** \brief Implementation of so_merge<N, M, 1, T> for se_perm<N, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_merge<N, M, 1, T>, se_perm<N - M + 1, T> > :
    public symmetry_operation_impl_base<
        so_merge<N, M, 1, T>, se_perm<N - M + 1, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<N, M, 1, T> operation_t;
    typedef se_perm<N - M + 1, T> element_t;
    typedef symmetry_operation_params<operation_t>
        symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

/** \brief Implementation of so_merge<N, N, 1, T> for se_perm<1, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_merge<N, N, 1, T>, se_perm<1, T> > :
    public symmetry_operation_impl_base<
        so_merge<N, N, 1, T>, se_perm<1, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<N, N, 1, T> operation_t;
    typedef se_perm<1, T> element_t;
    typedef symmetry_operation_params<operation_t>
        symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, size_t K, typename T>
const char *
symmetry_operation_impl< so_merge<N, M, K, T>, se_perm<N - M + K, T> >
::k_clazz =
	"symmetry_operation_impl< so_merge<N, M, K, T>, se_perm<N - M + K, T> >";

template<size_t N, typename T>
const char *
symmetry_operation_impl< so_merge<2 * N, 2 * N, N, T>, se_perm<N, T> >
::k_clazz =
    "symmetry_operation_impl< so_merge<2 * N, 2 * N, N, T>, se_perm<N, T> >";

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_merge<N, M, 1, T>, se_perm<N - M + 1, T> >
::k_clazz =
    "symmetry_operation_impl< so_merge<N, M, 1, T>, se_perm<N - M + 1, T> >";

template<size_t N, typename T>
const char *symmetry_operation_impl<
    so_merge<N, N, 1, T>, se_perm<1, T> >::k_clazz =
    "symmetry_operation_impl< so_merge<N, N, 1, T>, se_perm<1, T> >";

template<size_t N, size_t M, size_t K, typename T>
void
symmetry_operation_impl< so_merge<N, M, K, T>, se_perm<N - M + K, T> >
::do_perform(symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

	//	Adapter type for the input group
	//
    typedef se_perm<N, T> el1_t;
	typedef symmetry_element_set_adapter<N, T, el1_t> adapter1_t;
	typedef symmetry_element_set_adapter<N - M, T, element_t> adapter_t;

	//	Verify that the projection mask is correct
	//
	const mask<N> &m = params.msk;
	size_t nm = 0;
	for(size_t i = 0; i < N; i++) if(m[i]) nm++;

	if(nm != M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

	adapter1_t g1(params.grp1);
	permutation_group<N, T> grp1(g1);
	permutation_group<N - M, T> grpx;
	group1.stabilize(params.msk, groupx);

	permutation_group<M, T> group2;


	// 1) Permutation group w/o merged indexes

	// 2) Permutation group of merged indexes

	// 3) Permutation group of merged indexes that remain

	// 4) Extend the latter to a group of merged indexes

	// 5) Compute the intersection of groups from step 2) and 4)

	symmetry_element_set<N - M, T> set(se_perm<N - M, T>::k_sym_type);
	groupx.convert(set);

	adapterx_t gx(set);

	for (typename adapterx_t::iterator it = gx.begin(); it != gx.end(); it++) {

		const elx_t &ex = gx.get_elem(it);

		//	Projects the permutations onto a larger
		//	space and form a symmetry element
		sequence<N - M, size_t> a1(0);
		sequence<N - M + 1, size_t> a2a(0), a2b(0);
		for (size_t j = 0; j < N - M; j++) a1[j] = j;
		ex.get_perm().apply(a1);

		bool done = false;
		for(size_t j = 0, k = 0, l = 0; j < N; j++) {
			if (m[j]) {
				if (! done) {
					a2a[k] = N - M;
					a2b[k] = N - M;
					k++;
					done = true;
				}
				continue;
			}

			a2a[k] = l;
			a2b[k] = a1[l];
			k++;
			l++;
		}

		permutation_builder<N - M + 1> pb(a2b, a2a);
		se_perm<N - M + 1, T> e2(pb.get_perm(), ex.is_symm());
		params.grp2.insert(e2);

	}
}





template<size_t N, typename T>
void symmetry_operation_impl< so_merge<N, N, 1, T>,
	se_perm<1, T> >::do_perform(symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

	//	Verify that the projection mask is correct
	//
	const mask<N> &m = params.msk;
	size_t nm = 0;
	for(size_t i = 0; i < N; i++) if(m[i]) nm++;

	if(nm != N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}
}

} // namespace libtensor

// Template specializations
#include "so_merge_impl_perm_2n_2n_n.h"
#include "so_merge_impl_perm_n_m_1.h"
#include "so_merge_impl_perm_n_n_1.h"

#endif // LIBTENSOR_SO_MERGE_IMPL_PERM_H
