#ifndef LIBTENSOR_SYMMETRY_BASE_H
#define LIBTENSOR_SYMMETRY_BASE_H

#include "defs.h"
#include "exception.h"
#include "core/symmetry_i.h"
#include "core/orbit_iterator.h"

namespace libtensor {

template<size_t N, typename T, typename Sym> class symmetry_base;

template<size_t N, typename T, typename Sym>
class symmetry_target {
public:
	virtual bool is_same_impl(const Sym &other) const = 0;
};

template<size_t N, typename T>
class symmetry_target< N, T, symmetry_i<N, T> > {
public:
	virtual bool is_same_impl(const symmetry_i<N, T> &other) const {
		return false;
	}
};

template<size_t N, typename T, typename Parent>
class symmetry_aux {
public:
	template<typename Sym>
	static bool dispatch_is_same(const Sym *sym,
		const symmetry_i<N, T> &other) {

		typedef symmetry_base<N, T, Parent> parent_t;
		return static_cast<parent_t*>(sym)->
			symmetry_base<N, T, Parent>::is_same(other);
	}
};

template<size_t N, typename T>
class symmetry_aux< N, T, symmetry_i<N, T> > {
public:
	template<typename Sym>
	static bool dispatch_is_same(const Sym *sym,
		const symmetry_i<N, T> &other) {

		return false;
	}
};

template<size_t N, typename T, typename Sym>
class symmetry_base : virtual public symmetry_i<N, T> {
public:
	typedef symmetry_i<N, T> parent_t;

public:
	virtual bool is_same(const symmetry_i<N, T> &other) const;
};

template<size_t N, typename T, typename Sym>
bool symmetry_base<N, T, Sym>::is_same(const symmetry_i<N, T> &other) const {

	typedef symmetry_target<N, T, Sym> target_t;
	typedef typename Sym::parent_t parent_t;

	const target_t *t = dynamic_cast<const target_t*>(&other);
	if(t) {
		return t->is_same_impl(static_cast<const Sym&>(*this));
	} else {
		return symmetry_aux<N, T, parent_t>::dispatch_is_same(
			static_cast<const Sym*>(this), other);
	}
}

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_BASE_H
