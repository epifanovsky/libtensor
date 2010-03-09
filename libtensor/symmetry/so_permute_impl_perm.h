#ifndef LIBTENSOR_SO_PERMUTE_IMPL_PERM_H
#define LIBTENSOR_SO_PERMUTE_IMPL_PERM_H

#include <list>
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "permutation_group.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_i.h"
#include "symmetry_operation_impl.h"
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
	public symmetry_operation_impl_i {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_permute<N, M, T> operation_t;
	typedef se_perm<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

public:
	virtual const char *get_id() const {
		return element_t::k_sym_type;
	}

	virtual symmetry_operation_impl_i *clone() const {
		return new symmetry_operation_impl<operation_t, element_t>;
	}

	virtual void perform(symmetry_operation_params_i &params) const;

private:
	void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, typename T>
const char *symmetry_operation_impl<
	so_permute<N, T>, se_perm<N, T> >::k_clazz =
	"symmetry_operation_impl< so_permute<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_permute<N, T>, se_perm<N, T> >::perform(
	symmetry_operation_params_i &params) const {

	static const char *method = "perform(symmetry_operation_params_i&)";

	try {
		symmetry_operation_params_t &params2 =
			dynamic_cast<symmetry_operation_params_t&>(params);
		do_perform(params2);
	} catch(std::bad_cast&) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params: bad_cast");
	}
}


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