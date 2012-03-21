#ifndef LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_IMPL_H

#include "../bad_symmetry.h"

namespace libtensor {

template<size_t N, typename T>
const char *
symmetry_operation_impl< so_symmetrize<N, T>, se_label<N, T> >::k_clazz =
	"symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void
symmetry_operation_impl< so_symmetrize<N, T>, se_label<N, T> >::do_perform(
	symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(const symmetry_operation_params_t&)";

	typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter_t;

	adapter_t g1(params.grp1);
	params.grp2.clear();

	sequence<N, size_t> map;
	for (register size_t i = 0; i < N; i++) map[i] = i;

	for(typename adapter_t::iterator it = g1.begin(); it != g1.end(); it++) {

        const se_label<N, T> &e1 = g1.get_elem(it);
		const block_labeling<N> &bl1 = e1.get_labeling();
		const evaluation_rule<N> &r1 = e1.get_rule();

		register size_t i = 0;
		for (; i < N && ! params.msk[i]; i++) ;

		size_t typei = bl1.get_dim_type(i);
		i++;
		for (; i < N; i++) {
		    if (! params.msk[i]) continue;
		    if (bl1.get_dim_type(i) != typei)
		        throw bad_symmetry(g_ns, k_clazz, method,
		                __FILE__, __LINE__, "Incompatible dimensions.");
		}

		se_label<N, T> e2(bl1.get_block_index_dims(), e1.get_table_id());
		block_labeling<N> &bl2 = e2.get_labeling();
		transfer_labeling(bl1, map, bl2);
		evaluation_rule<N> r2(r1);
		r2.symmetrize(params.msk);
		e2.set_rule(r2);

		params.grp2.insert(e2);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_LABEL_IMPL_H
