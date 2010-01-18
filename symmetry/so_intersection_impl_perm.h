#ifndef LIBTENSOR_SO_INTERSECTION_IMPL_PERM_H
#define LIBTENSOR_SO_INTERSECTION_IMPL_PERM_H

#include "symmetry_element_set_adapter.h"
#include "so_intersection.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_intersection<N, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class so_intersection_impl< se_perm<N, T> > {
public:
	static const char *k_clazz; //!< Class name

public:
	void perform(const symmetry_operation_params<
		so_intersection<N, T> > &params,
		symmetry_element_set<N, T> &set);
};


template<size_t N, typename T>
const char *so_intersection_impl< se_perm<N, T> >::k_clazz =
	"so_intersection_impl< se_perm<N, T> >";


template<size_t N, typename T>
void so_intersection_impl< se_perm<N, T> >::perform(
	const symmetry_operation_params< so_intersection<N, T> > &params,
	symmetry_element_set<N, T> &set) {

	static const char *method =
		"perform(const symmetry_operation_params< "
		"so_intersection<N, T> >&, symmetry_element_set<N, T>&)";

	typedef symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter_t;
	adapter_t g1(params.g1);
	adapter_t g2(params.g2);
	typename adapter_t::iterator i, j;

	bool g1_empty = g1.is_empty(), g2_empty = g2.is_empty();

	//
	//	Empty the output set
	//
	set.clear();

	//
	//	When G1=0 or G2=0, G1 \cap G2 = 0
	//
	if(g1_empty || g2_empty) return;

	//
	//	When G1!=0 and G2!=0 are non-overlapping, G1 \cap G2 = 0
	//
	mask<N> m0, m1, m2;
	for(i = g1.begin(); i != g1.end(); i++) m1 |= g1.get_elem(i).get_mask();
	for(i = g2.begin(); i != g2.end(); i++) m2 |= g2.get_elem(i).get_mask();
	if((m1 & m2).equals(m0)) return;

	//
	//	When G1=G2, G1 \cap G2 = G1
	//
	bool g1_eq_g2 = true;
	for(i = g1.begin(); i != g1.end(); i++) {
		const se_perm<N, T> &e1 = g1.get_elem(i);
		size_t neq = 0;
		for(j = g2.begin(); j != g2.end(); j++) {
			const se_perm<N, T> &e2 = g2.get_elem(j);
			if(e1.get_perm().equals(e2.get_perm())) neq++;
		}
		if(neq == 0) {
			g1_eq_g2 = false;
			break;
		}
		if(neq > 1) {
			throw bad_symmetry(g_ns, k_clazz, method, __FILE__,
				__LINE__, "Repetitive elements in g2.");
		}
	}
	if(g1_eq_g2) {
		for(i = g1.begin(); i != g1.end(); i++)
			set.insert(g1.get_elem(i));
		return;
	}

	throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
		"Unhandled case.");
}


} // namespace libtensor

#endif // LIBTENSOR_SO_INTERSECTION_IMPL_PERM_H

