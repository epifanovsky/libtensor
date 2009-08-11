#ifndef LIBTENSOR_SYMMETRY_ELEMENT_BASE_H
#define LIBTENSOR_SYMMETRY_ELEMENT_BASE_H

#include "defs.h"
#include "exception.h"
#include "core/symmetry_element_i.h"
#include "symmetry_element_target.h"

namespace libtensor {


template<size_t N, typename T, typename Parent>
class symmetry_element_aux {
public:
	template<typename ElemT>
	static void dispatch(const ElemT *elem,
		symmetry_element_target_i<N, T> &tgt) {

		typedef symmetry_element_base<N, T, Parent> parent_t;
		static_cast<const parent_t*>(elem)->
			symmetry_base<N, T, Parent>::dispatch(tgt);
	}

};


template<size_t N, typename T>
class symmetry_element_aux< N, T, symmetry_element_i<N, T> > {
public:
	template<typename ElemT>
	static void dispatch(const ElemT *elem,
		symmetry_element_target_i<N, T> &tgt) {

		tgt.accept_default(*elem);
	}

};


template<size_t N, typename T, typename ElemT>
class symmetry_element_base : virtual public symmetry_element_i<N, T> {
public:
	typedef symmetry_element_i<N, T> parent_t;

public:
	virtual void dispatch(symmetry_element_target_i<N, T> &tgt) const
		throw(exception);

};


template<size_t N, typename T, typename ElemT>
void symmetry_element_base<N, T, ElemT>::dispatch(
	symmetry_element_target_i<N, T> &tgt) const throw(exception) {

	typedef symmetry_element_target<N, T, ElemT> target_t;
	typedef typename ElemT::parent_t parent_t;

	target_t *t = dynamic_cast<target_t*>(&tgt);
	if(t) {
		t->accept(static_cast<const ElemT&>(*this));
	} else {
		symmetry_element_aux<N, T, parent_t>::dispatch(
			static_cast<const ElemT*>(this), tgt);
	}
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_ELEMENT_BASE_H
