#ifndef LIBTENSOR_SO_SYMMETRIZE_IMPL_LABEL_H
#define LIBTENSOR_SO_SYMMETRIZE_IMPL_LABEL_H

#include "../defs.h"
#include "../exception.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_symmetrize.h"
#include "se_label.h"

namespace libtensor {


/**	\brief Implementation of so_symmetrize<N, T> for se_label<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_symmetrize<N, T>, se_label<N, T> > :
	public symmetry_operation_impl_base< so_symmetrize<N, T>,
		se_label<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_symmetrize<N, T> operation_t;
	typedef se_label<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_symmetrize<N, T>,
	se_label<N, T> >::k_clazz =
	"symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_symmetrize<N, T>, se_label<N, T> >::do_perform(
	symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(const symmetry_operation_params_t&)";

	typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter_t;

	adapter_t g1(params.grp1);
	params.grp2.clear();

	size_t map[N];
	for (size_t j = 0; j < N; j++) map[j] = j;
	params.perm.apply(map);


	for(typename adapter_t::iterator i = g1.begin(); i != g1.end(); i++) {

		const se_label<N, T> &e1 = g1.get_elem(i);

		for (size_t j = 0; j < N; j++) {
			if (map[j] > j) {

				if (e1.get_dim_type(j) != e1.get_dim_type(map[j]))
					throw bad_symmetry(g_ns, k_clazz, method,
						__FILE__, __LINE__, "Incompatible dimensions.");
			}
		}

		params.grp2.insert(e1);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_IMPL_LABEL_H
