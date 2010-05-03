#ifndef LIBTENSOR_SO_UNION_IMPL_PERM_H
#define LIBTENSOR_SO_UNION_IMPL_PERM_H

#include "permutation_group.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_i.h"
#include "symmetry_operation_impl.h"
#include "so_union.h"
#include "se_perm.h"

namespace libtensor {


/**	\brief Implementation of so_union<N, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_union<N, T>, se_perm<N, T> > :
	public symmetry_operation_impl_i {
public:
	static const char *k_clazz; //!< Class name

public:
	typedef so_union<N, T> operation_t;
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
const char *symmetry_operation_impl< so_union<N, T>, se_perm<N, T> >::k_clazz =
	"symmetry_operation_impl< so_union<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_union<N, T>, se_perm<N, T> >::perform(
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
void symmetry_operation_impl< so_union<N, T>, se_perm<N, T> >::do_perform(
	symmetry_operation_params< so_union<N, T> > &params) const {

	static const char *method =
		"perform(symmetry_operation_params< so_union<N, T> >&)";

/*
	typedef symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter_t;
	adapter_t g1(params.g1);
	adapter_t g2(params.g2);
	typename adapter_t::iterator i, j;

	bool g1_empty = g1.is_empty(), g2_empty = g2.is_empty();

	//
	//	When G1=0 and G2=0, G1 \cup G2 = 0
	//
	if(g1_empty && g2_empty) return;

	//
	//	When G1=0 and G2!=0, G1 \cup G2 = G2
	//	When G1!=0 and G2=0, G1 \cup G2 = G1
	//
	if(g1_empty || g2_empty) {

		adapter_t &g = g1_empty ? g2 : g1;
		for(typename adapter_t::iterator i = g.begin();
			i != g.end(); i++) {

			set.insert(g.get_elem(i));
		}
		return;
	}

	//
	//	When G1!=0 and G2!=0 are non-overlapping, their union is
	//	their combination
	//
	mask<N> m0, m1, m2;
	for(i = g1.begin(); i != g1.end(); i++) m1 |= g1.get_elem(i).get_mask();
	for(i = g2.begin(); i != g2.end(); i++) m2 |= g2.get_elem(i).get_mask();
	if((m1 & m2).equals(m0)) {
		for(i = g1.begin(); i != g1.end(); i++)
			set.insert(g1.get_elem(i));
		for(i = g2.begin(); i != g2.end(); i++)
			set.insert(g2.get_elem(i));
		return;
	}

	//
	//	When G1=G2!=0, G1 \cup G2 = G1
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
		*/
}


} // namespace libtensor

#endif // LIBTENSOR_SO_UNION_IMPL_PERM_H

