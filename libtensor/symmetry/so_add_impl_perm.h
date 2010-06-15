#ifndef LIBTENSOR_SO_ADD_IMPL_PERM_H
#define LIBTENSOR_SO_ADD_IMPL_PERM_H

#include "../defs.h"
#include "../exception.h"
#include "permutation_group.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_add.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_add<N, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_add<N, T>, se_perm<N, T> > :
	public symmetry_operation_impl_base< so_add<N, T>, se_perm<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_add<N, T> operation_t;
	typedef se_perm<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_add<N, T>, se_perm<N, T> >::k_clazz =
	"symmetry_operation_impl< so_add<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_add<N, T>, se_perm<N, T> >::do_perform(
	symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(const symmetry_operation_params_t&)";

	typedef symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter_t;
	adapter_t adapter1(params.grp1);
	adapter_t adapter2(params.grp2);

	permutation_group<N, T> grp1(adapter1);
	permutation_group<N, T> grp3;
	for(typename adapter_t::iterator i = adapter2.begin();
		i != adapter2.end(); i++) {

		const se_perm<N, T> &e = adapter2.get_elem(i);
		if(grp1.is_member(e.is_symm(), e.get_perm())) {
			grp3.add_orbit(e.is_symm(), e.get_perm());
		}
	}

	params.grp3.clear();
	grp3.convert(params.grp3);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_ADD_IMPL_PERM_H
