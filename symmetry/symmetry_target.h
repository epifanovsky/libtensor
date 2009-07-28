#ifndef LIBTENSOR_SYMMETRY_TARGET_H
#define LIBTENSOR_SYMMETRY_TARGET_H

#include "defs.h"
#include "exception.h"
#include "core/symmetry_i.h"
#include "core/symmetry_target_i.h"

namespace libtensor {

template<size_t N, typename T>
class symmetry_target_base : public symmetry_target_i<N, T> {
public:
	//!	\name Implementation of libtensor::symmetry_target_i<N, T>
	//@{
	virtual void accept_default(const symmetry_i<N, T> &sym)
		throw(exception) { };
	virtual void accept_default(symmetry_i<N, T> &sym) throw(exception) { };
	//@}

};

template<size_t N, typename T, typename Sym>
class symmetry_target : virtual public symmetry_target_base<N, T> {
public:
	virtual void accept(Sym &sym) throw(exception) { };
};

template<size_t N, typename T, typename Sym>
class symmetry_const_target : virtual public symmetry_target_base<N, T> {
public:
	virtual void accept(const Sym &sym) throw(exception) { };
};

template<size_t N, typename T>
class symmetry_target< N, T, symmetry_i<N, T> > :
	virtual public symmetry_target_base<N, T> {

public:
	virtual void accept(symmetry_i<N, T> &sym) throw(exception) {
		accept_default(sym);
	};
};

template<size_t N, typename T>
class symmetry_const_target< N, T, symmetry_i<N, T> > :
	virtual public symmetry_target_base<N, T> {

public:
	virtual void accept(const symmetry_i<N, T> &sym) throw(exception) {
		accept_default(sym);
	};
};

} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_TARGET_H
