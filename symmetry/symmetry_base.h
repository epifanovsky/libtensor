#ifndef LIBTENSOR_SYMMETRY_BASE_H
#define LIBTENSOR_SYMMETRY_BASE_H

#include "defs.h"
#include "exception.h"
#include "core/symmetry_i.h"
#include "symmetry_target.h"

namespace libtensor {

template<size_t N, typename T, typename Sym> class symmetry_base;

template<size_t N, typename T, typename Parent>
class symmetry_aux {
public:
	template<typename Sym>
	static void dispatch(const Sym *sym, symmetry_target_i<N, T> &target) {

		typedef symmetry_base<N, T, Parent> parent_t;
		static_cast<const parent_t*>(sym)->
			symmetry_base<N, T, Parent>::dispatch(target);
	}

	template<typename Sym>
	static void dispatch(Sym *sym, symmetry_target_i<N, T> &target) {

		typedef symmetry_base<N, T, Parent> parent_t;
		static_cast<parent_t*>(sym)->
			symmetry_base<N, T, Parent>::dispatch(target);
	}
};

template<size_t N, typename T>
class symmetry_aux< N, T, symmetry_i<N, T> > {
public:
	template<typename Sym>
	static void dispatch(const Sym *sym, symmetry_target_i<N, T> &target) {

		target.accept_default(*sym);
	}

	template<typename Sym>
	static void dispatch(Sym *sym, symmetry_target_i<N, T> &target) {

		target.accept_default(*sym);
	}

};

template<size_t N, typename T, typename Sym>
class symmetry_base : virtual public symmetry_i<N, T> {
public:
	typedef symmetry_i<N, T> parent_t;

public:
	virtual void dispatch(symmetry_target_i<N, T> &target) const
		throw(exception);
	virtual void dispatch(symmetry_target_i<N, T> &target)
		throw(exception);

};

template<size_t N, typename T, typename Sym>
void symmetry_base<N, T, Sym>::dispatch(symmetry_target_i<N, T> &target)
	const throw(exception) {

	typedef symmetry_const_target<N, T, Sym> target_t;
	typedef typename Sym::parent_t parent_t;

	target_t *t = dynamic_cast<target_t*>(&target);
	if(t) {
		t->accept(static_cast<const Sym&>(*this));
	} else {
		symmetry_aux<N, T, parent_t>::dispatch(
			static_cast<const Sym*>(this), target);
	}
}

template<size_t N, typename T, typename Sym>
void symmetry_base<N, T, Sym>::dispatch(symmetry_target_i<N, T> &target)
	throw(exception) {

	typedef symmetry_target<N, T, Sym> target_t;
	typedef symmetry_const_target<N, T, Sym> const_target_t;
	typedef typename Sym::parent_t parent_t;

	target_t *t = dynamic_cast<target_t*>(&target);
	if(t) {
		t->accept(static_cast<Sym&>(*this));
	} else {
		const_target_t *t1 = dynamic_cast<const_target_t*>(&target);
		if(t1) {
			t1->accept(static_cast<const Sym&>(*this));
		} else {
			symmetry_aux<N, T, parent_t>::dispatch(
				static_cast<Sym*>(this), target);
		}
	}
}

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_BASE_H
