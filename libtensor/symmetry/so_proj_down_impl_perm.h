#ifndef LIBTENSOR_SO_PROJ_DOWN_IMPL_PERM_H
#define LIBTENSOR_SO_PROJ_DOWN_IMPL_PERM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "permutation_group.h"
#include "symmetry_element_set_adapter.h"
#include "so_proj_down.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_proj_down<N, M, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_proj_down_impl< se_perm<N, T> > {
public:
	static const char *k_clazz; //!< Class name

private:
	struct subgroup {
		permutation<N> perm;
		mask<N> msk;
		sequence<N, size_t> cycles;
		bool sign; // T=symmetric/F=antisymmetric
		bool sym; // cyclic/symmetric
		subgroup() : cycles(0), sign(false), sym(false) { }
		subgroup(const permutation<N> &perm_, const mask<N> &msk_,
			const sequence<N, size_t> &cycles_, bool sign_,
			bool sym_) : perm(perm_), msk(msk_), cycles(cycles_),
			sign(sign_), sym(sym_) { }
	};

public:
	template<size_t M>
	void perform(
		const symmetry_operation_params<
			so_proj_down<N, M, T> > &params,
		symmetry_element_set<N - M, T> &set);

};


template<size_t N, typename T>
const char *so_proj_down_impl< se_perm<N, T> >::k_clazz =
	"so_proj_down_impl< se_perm<N, T> >";


template<size_t N, typename T> template<size_t M>
void so_proj_down_impl< se_perm<N, T> >::perform(
	const symmetry_operation_params< so_proj_down<N, M, T> > &params,
	symmetry_element_set<N - M, T> &set) {

	static const char *method =
		"perform<M>(const symmetry_operation_params< "
		"so_proj_down<N, M, T> >&, symmetry_element_set<N - M, T>&)";

	//	Adapter type for the input group
	//
	typedef symmetry_element_set_adapter< N, T, se_perm<N, T> >
		adapter_t;

	//	Verify that the projection mask is correct
	//
	const mask<N> &m = params.msk;
	size_t nm = 0;
	for(size_t i = 0; i < N; i++) if(m[i]) nm++;
	if(nm != N - M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}
	if(!params.perm.is_identity()) {
		throw not_implemented(g_ns, k_clazz, method, __FILE__, __LINE__);
	}

	adapter_t adapter1(params.grp);
	permutation_group<N, T> group1(adapter1);
	permutation_group<N - M, T> group2;
	group1.project_down(params.msk, group2);
	group2.convert(set);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_DOWN_IMPL_PERM_H