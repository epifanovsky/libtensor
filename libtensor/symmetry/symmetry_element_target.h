#ifndef LIBTENSOR_SYMMETRY_ELEMENT_TARGET_H
#define LIBTENSOR_SYMMETRY_ELEMENT_TARGET_H

#include "../defs.h"
#include "../exception.h"
#include "../core/symmetry_element_i.h"

namespace libtensor {


template<size_t N, typename T>
class symmetry_element_target_base : public symmetry_element_target_i<N, T> {
public:
	//!	\name Implementation of
	//!		libtensor::symmetry_element_target_i<N, T>
	//@{
	virtual void accept_default(const symmetry_element_i<N, T> &elem)
		throw(exception) { };
	//@}

};


template<size_t N, typename T, typename ElemT>
class symmetry_element_target :
	virtual public symmetry_element_target_base<N, T> {

public:
	virtual void accept(const ElemT &elem) throw(exception) { };
};


template<size_t N, typename T>
class symmetry_element_target< N, T, symmetry_element_i<N, T> > :
	virtual public symmetry_element_target_base<N, T> {

public:
	virtual void accept(const symmetry_element_i<N, T> &elem)
		throw(exception) {

		accept_default(elem);
	};

};


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_ELEMENT_TARGET_H
