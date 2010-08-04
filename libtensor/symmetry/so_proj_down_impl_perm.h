#ifndef LIBTENSOR_SO_PROJ_DOWN_IMPL_PERM_H
#define LIBTENSOR_SO_PROJ_DOWN_IMPL_PERM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "permutation_group.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_proj_down.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_proj_down<N, M, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	The implementation stabilizes unmasked dimensions pointwise.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_proj_down<N, M, T>, se_perm<N, T> > :
	public symmetry_operation_impl_base<
		so_proj_down<N, M, T>, se_perm<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_proj_down<N, M, T> operation_t;
	typedef se_perm<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl<
	so_proj_down<N, M, T>, se_perm<N, T> >::k_clazz =
	"symmetry_operation_impl< so_proj_down<N, M, T>, se_perm<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_proj_down<N, M, T>,
	se_perm<N, T> >::do_perform(symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

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

	adapter_t adapter1(params.grp1);
	permutation_group<N, T> group1(adapter1);
	permutation_group<N - M, T> group2;
	group1.project_down(params.msk, group2);
	group2.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_DOWN_IMPL_PERM_H
