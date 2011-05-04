#ifndef LIBTENSOR_SO_APPLY_IMPL_PERM_H
#define LIBTENSOR_SO_APPLY_IMPL_PERM_H

#include "../defs.h"
#include "../exception.h"
#include "permutation_group.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_apply.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_add<N, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> > :
	public symmetry_operation_impl_base< so_apply<N, T>, se_perm<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_apply<N, T> operation_t;
	typedef se_perm<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> >::k_clazz =
	"symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> >::do_perform(
	symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(const symmetry_operation_params_t&)";

	//	Adapter type for the input group
	//
	typedef se_perm<N, T> se_perm_t;
	typedef symmetry_element_set_adapter<N, T, se_perm_t> adapter_t;

	params.grp2.clear();

	// Result symmetry element set is empty, if functor is asymmetric
	if (params.is_asym) return;

	adapter_t adapter1(params.grp1);
	permutation_group<N, T> group(adapter1);
	group.permute(params.perm1);

	// If functor is symmetric
	if (params.sign) {
		symmetry_element_set<N, T> set(se_perm_t::k_sym_type);
		group.convert(set);

		adapter_t adapter(set);
		for (typename adapter_t::iterator it = adapter.begin();
				it != adapter.end(); it++) {

			const se_perm_t &el = adapter.get_elem(it);
			params.grp2.insert(se_perm_t(el.get_perm(), true));
		}
	}
	else {
		group.convert(params.grp2);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_IMPL_PERM_H
