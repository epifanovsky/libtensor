#ifndef LIBTENSOR_SO_PROJ_UP_IMPL_PERM_H
#define LIBTENSOR_SO_PROJ_UP_IMPL_PERM_H

#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "symmetry_element_set_adapter.h"
#include "so_proj_up.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_proj_up<N, M, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_proj_up_impl< se_perm<N, T> > {
public:
	static const char *k_clazz; //!< Class name

public:
	template<size_t M>
	void perform(
		const symmetry_operation_params< so_proj_up<N, M, T> > &params,
		symmetry_element_set<N + M, T> &set);

private:
};


template<size_t N, typename T>
const char *so_proj_up_impl< se_perm<N, T> >::k_clazz =
	"so_proj_up_impl< se_perm<N, T> >";


template<size_t N, typename T> template<size_t M>
void so_proj_up_impl< se_perm<N, T> >::perform(
	const symmetry_operation_params< so_proj_up<N, M, T> > &params,
	symmetry_element_set<N + M, T> &set) {

	static const char *method =
		"perform<M>(const symmetry_operation_params< "
		"so_proj_up<N, M, T> >&, symmetry_element_set<N + M, T>&)";

	//	Adapter type for the input group
	typedef symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter_t;

	//	Verify that the mask is valid
	const mask<N + M> &m = params.msk;
	size_t nm = 0;
	for(size_t i = 0; i < N + M; i++) if(m[i]) nm++;
	if(nm != N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

	//	Non-identity permutations are not supported yet
	//~ if(!params.perm.is_identity()) {
		//~ throw not_implemented(g_ns, k_clazz, method,
			//~ __FILE__, __LINE__);
	//~ }

	adapter_t g1(params.grp);

	//	Go over each element in the source group and project
	for(typename adapter_t::iterator i = g1.begin(); i != g1.end(); i++) {

		const se_perm<N, T> &e1 = g1.get_elem(i);

		//	Analyze permutations
		//
		//	The permutation of the cycle (params.perm) is only
		//	allowed to alter the cycle, but not change its order.
		//
		//~ size_t order1 = 0, order2 = 0;
		//~ permutation<N> p1a(e1.get_perm()), p1b(params.perm);
		//~ p1b.permute(p1a);
		//~ permutation<N> p1c(p1b);
		//~ while(!p1a.is_identity()) {
			//~ p1a.permute(e1.get_perm());
			//~ order1++;
		//~ }
		//~ while(!p1c.is_identity()) {
			//~ p1c.permute(p1b);
			//~ order2++;
		//~ }
		//~ if(order1 != order2) {
			//~ set.clear();
			//~ throw bad_parameter(g_ns, k_clazz, method,
				//~ __FILE__, __LINE__, "params.perm");
		//~ }

		//	Project the combined permutation onto the larger
		//	space and form a symmetry element
		size_t a1[N], a2a[N + M], a2b[N + M];
		size_t b1[N], b2a[N + M], b2b[N + M];
		size_t k = 0;
		for(size_t j = 0; j < N; j++) a1[j] = b1[j] = j;
		e1.get_perm().apply(a1);
		params.perm.apply(b1);
		for(size_t j = 0; j < N + M; j++) {
			if(m[j]) {
				a2a[j] = k; a2b[j] = a1[k];
				b2a[j] = k; b2b[j] = b1[k];
				k++;
			}
		}
		for(size_t j = 0; j < N + M; j++) {
			if(!m[j]) {
				a2a[j] = a2b[j] = k;
				b2a[j] = b2b[j] = k;
				k++;
			}
		}

		permutation_builder<N + M> pb_map(b2b, b2a);
		permutation_builder<N + M> pb(a2a, a2b, pb_map.get_perm());
		se_perm<N + M, T> e2(pb.get_perm(), e1.is_symm());
		set.insert(e2);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_UP_IMPL_PERM_H