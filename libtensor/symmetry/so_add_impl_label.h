#ifndef LIBTENSOR_SO_ADD_IMPL_LABEL_H
#define LIBTENSOR_SO_ADD_IMPL_LABEL_H

#include "../defs.h"
#include "../exception.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_add.h"
#include "se_label.h"

namespace libtensor {


/**	\brief Implementation of so_add<N, T> for se_label<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_add<N, T>, se_label<N, T> > :
	public symmetry_operation_impl_base< so_add<N, T>, se_label<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_add<N, T> operation_t;
	typedef se_label<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_add<N, T>, se_label<N, T> >::k_clazz =
	"symmetry_operation_impl< so_add<N, T>, se_label<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_add<N, T>, se_label<N, T> >::do_perform(
	symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(const symmetry_operation_params_t&)";

	typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter_t;
	adapter_t adapter1(params.grp1);
	adapter_t adapter2(params.grp2);
	params.grp3.clear();

	for (typename adapter_t::iterator it1 = adapter1.begin();
			it1 != adapter1.end(); it1++) {

		std::string id1(adapter1.get_elem(it1).get_table_id());

		typename adapter_t::iterator it2 = adapter2.begin();
		for(; it2 != adapter2.end(); it2++) {

			if (id1.compare(adapter2.get_elem(it2).get_table_id()) == 0)
				break;
		}

		if (it2 == adapter2.end()) continue;

		// if targets are different we don't need an se_label element
		// in the result
		if (adapter1.get_elem(it1).get_target()
				!= adapter2.get_elem(it2).get_target()) continue;


#ifdef LIBTENSOR_DEBUG
		se_label<N, T> se1(adapter1.get_elem(it1)),
				se2(adapter2.get_elem(it2));
		se1.permute(params.perm1);
		se2.permute(params.perm2);

		for (size_t i = 0; i < N; i++) {
			size_t dim = se1.get_dim(i);
			for (size_t j = 0; j < dim; j++) {
				if (se1.get_label(i, j) != se2.get_label(i, j))
					throw bad_symmetry(g_ns, k_clazz, method,
							__FILE__, __LINE__, "Incompatible labeling.");
			}
		}
#endif

		// create new se_label element
		se_label<N, T> se3(adapter1.get_elem(it1));
		se3.permute(params.perm1);

		params.grp3.insert(se3);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_ADD_IMPL_PERM_H
