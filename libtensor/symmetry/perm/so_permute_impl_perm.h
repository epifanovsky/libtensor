#ifndef LIBTENSOR_SO_PERMUTE_IMPL_PERM_H
#define LIBTENSOR_SO_PERMUTE_IMPL_PERM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "permutation_group.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_permute.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_permute<N, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_permute<N, T>, se_perm<N, T> > :
	public symmetry_operation_impl_base< so_permute<N, T>, se_perm<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_permute<N, T> operation_t;
	typedef se_perm<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, typename T>
const char *symmetry_operation_impl<
	so_permute<N, T>, se_perm<N, T> >::k_clazz =
	"symmetry_operation_impl< so_permute<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_permute<N, T>, se_perm<N, T> >::do_perform(
	symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

	//	Adapter type for the input group
	//
	typedef symmetry_element_set_adapter< N, T, se_perm<N, T> >
		adapter_t;

	adapter_t adapter1(params.grp1);
	permutation_group<N, T> group(adapter1);
	group.permute(params.perm);
	group.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PERMUTE_IMPL_PERM_H
