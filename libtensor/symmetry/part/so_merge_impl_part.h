#ifndef LIBTENSOR_SO_MERGE_IMPL_PART_H
#define LIBTENSOR_SO_MERGE_IMPL_PART_H

#include <list>
#include "../core/block_index_subspace_builder.h"
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "partition_set.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "so_merge.h"
#include "se_part.h"

namespace libtensor {


/**	\brief Implementation of so_merge<N, M, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	The implementation stabilizes unmasked dimensions pointwise.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_merge<N, M, T>, se_part<N, T> > :
	public symmetry_operation_impl_base<
		so_merge<N, M, T>, se_part<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_merge<N, M, T> operation_t;
	typedef se_part<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl<
	so_merge<N, M, T>, se_part<N, T> >::k_clazz =
	"symmetry_operation_impl< so_merge<N, M, T>, se_part<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_merge<N, M, T>,
	se_part<N, T> >::do_perform(symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

	//	Adapter type for the input group
	//
	typedef symmetry_element_set_adapter<N, T, element_t> adapter1_t;
	typedef se_part<N - M + 1, T> elx_t;
	typedef symmetry_element_set_adapter<N - M + 1, T, elx_t> adapterx_t;

	//	Verify that the projection mask is correct
	//
	const mask<N> &m = params.msk;
	size_t nm = 0;
	for(size_t i = 0; i < N; i++) if(m[i]) nm++;

	if(nm != M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

	adapter1_t g1(params.grp1);
	params.grp2.clear();

	if (g1.is_empty()) return;

	// mask of remaining dimensions
	mask<N> mr;
	bool done = false;
	for (size_t i = 0; i < N; i++) {
		if (m[i] && done) continue;
		if (m[i]) done = true;
		mr[i] = true;
	}

	// create block index dimensions of result
	typename adapter1_t::iterator it = g1.begin();
	block_index_subspace_builder<N - M + 1, M - 1> rbb(
			g1.get_elem(it).get_bis(), mr);

	// merge the partitions
	partition_set<N, T> pset1(g1);
	partition_set<N - M + 1, T> pset2(rbb.get_bis());
	pset1.merge(m, pset2);

	pset2.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_PERM_H
