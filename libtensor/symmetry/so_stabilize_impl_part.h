#ifndef LIBTENSOR_SO_STABILIZE_IMPL_PART_H
#define LIBTENSOR_SO_STABILIZE_IMPL_PART_H

#include "../core/block_index_subspace_builder.h"
#include "../defs.h"
#include "../exception.h"
#include "partition_set.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_stabilize.h"
#include "se_part.h"

namespace libtensor {


/**	\brief Implementation of so_stabilize<N, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_impl< so_stabilize<N, M, K, T>, se_part<N, T> > :
	public symmetry_operation_impl_base< so_stabilize<N, M, K, T>, se_part<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_stabilize<N, M, K, T> operation_t;
	typedef se_part<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, size_t M, size_t K, typename T>
const char *symmetry_operation_impl< so_stabilize<N, M, K, T>, se_part<N, T> >::k_clazz =
	"symmetry_operation_impl< so_stabilize<N, M, K, T>, se_part<N, T> >";

template<size_t N, size_t M, size_t K, typename T>
void symmetry_operation_impl< so_stabilize<N, M, K, T>, se_part<N, T> >::do_perform(
	symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(const symmetry_operation_params_t&)";

	typedef symmetry_element_set_adapter< N, T, element_t> adapter_t;

	//	Verify that the projection mask is correct
	//
	mask<N> tm, rm;
	size_t m = 0;
	for (size_t k = 0; k < K; k++) {
		const mask<N> &msk = params.msk[k];
		for(size_t i = 0; i < N; i++) {
			if(! msk[i]) continue;
			if(tm[i])
				throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
					"params.msk[k]");

			tm[i] = true;
			m++;
		}
	}
	if(m != M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

	for (size_t i = 0; i < N; i++) rm[i] = ! tm[i];

	adapter_t g1(params.grp1);
	params.grp2.clear();

	typename adapter_t::iterator it = g1.begin();
	if (it == g1.end()) return;

	block_index_subspace_builder<N - M, M> bb(g1.get_elem(it).get_bis(), rm);

	partition_set<N, T> ps1(g1);
	partition_set<N - M, T> ps2(bb.get_bis());
	ps1.stabilize(params.msk, ps2);

	ps2.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_STABILIZE_IMPL_PART_H
