#ifndef LIBTENSOR_SO_CONCAT_IMPL_PART_H
#define LIBTENSOR_SO_CONCAT_IMPL_PART_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../not_implemented.h"
#include "../../core/permutation_builder.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_concat.h"
#include "../se_part.h"
#include "partition_set.h"

namespace libtensor {


/**	\brief Implementation of so_concat<N, M, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_concat<N, M, T>, se_part<N, T> > :
	public symmetry_operation_impl_base<
		so_concat<N, M, T>, se_part<N, T> > {

public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_concat<N, M, T> operation_t;
	typedef se_part<N, T> element_t;
	typedef symmetry_operation_params<operation_t>
		symmetry_operation_params_t;

protected:
	virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl< so_concat<N, M, T>,
	se_part<N, T> >::k_clazz =
	"symmetry_operation_impl< so_concat<N, M, T>, se_part<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_concat<N, M, T>,
	se_part<N, T> >::do_perform(symmetry_operation_params_t &params) const {

	static const char *method = "do_perform(symmetry_operation_params_t&)";

	// Adapter type for the input groups
	typedef symmetry_element_set_adapter< N, T, se_part<N, T> > adapter1_t;
	typedef symmetry_element_set_adapter< M, T, se_part<M, T> > adapter2_t;

	adapter1_t g1(params.g1);
	adapter2_t g2(params.g2);
	params.g3.clear();

	// map result index to input index
	sequence<N + M, size_t> map(0);
	for (size_t i = 0; i < N + M; i++) map[i] = i;
	params.perm.apply(map);

	mask<N + M> msk1, msk2;
	sequence<N, size_t> seq1a(0), seq1b(0);
	sequence<M, size_t> seq2a(0), seq2b(0);
	for (size_t i = 0, j = 0, k = 0; i < N + M; i++) {
		if (map[i] < N) {
			msk1[i] = true;
			seq1a[j] = j; seq1b[j] = map[i];
			j++;
		}
		else {
			msk2[i] = true;
			seq2a[k] = k; seq2b[k] = map[i] - N;
			k++;
		}
	}
	permutation_builder<N> pb1(seq1b, seq1a);
	permutation_builder<M> pb2(seq2b, seq2a);

	partition_set<N + M, double> pset1(params.bis);

	//	Go over each element in the first source group
	for(typename adapter1_t::iterator it1 = g1.begin();
			it1 != g1.end(); it1++) {

		pset1.add_partition(g1.get_elem(it1), pb1.get_perm(), msk1);
	}

	if (params.dirsum) {
	    partition_set<N + M, double> pset2(params.bis);
	    // Go over each element in the second source group
	    for(typename adapter2_t::iterator it2 = g2.begin();
	            it2 != g2.end(); it2++) {

	        pset2.add_partition(g2.get_elem(it2), pb2.get_perm(), msk2);
	    }

	    pset1.intersect(pset2, false);
	}
	else {
        // Go over each element in the second source group
        for(typename adapter2_t::iterator it2 = g2.begin();
                it2 != g2.end(); it2++) {

            pset1.add_partition(g2.get_elem(it2), pb2.get_perm(), msk2);
        }
	}

	pset1.convert(params.g3);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_CONCAT_IMPL_PART_H
