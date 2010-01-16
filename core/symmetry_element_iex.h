#ifndef LIBTENSOR_SYMMETRY_ELEMENT_IEX_H
#define LIBTENSOR_SYMMETRY_ELEMENT_IEX_H

#include "symmetry_element_i.h"

namespace libtensor {


template<size_t N, typename T>
class symmetry_element_target_i;


template<size_t N, typename T>
class symmetry_element_iex : public symmetry_element_i<N, T> {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Virtual destructor
	 **/
	virtual ~symmetry_element_iex() { };

	//@}


	//!	\name Interface symmetry_element_iex<N, T>
	//@{

	/**	\brief Checks whether two %symmetry elements are equal
			(have the same type and perform identically)
	 **/
	virtual bool equals(const symmetry_element_i<N, T> &se) const = 0;

	virtual void dispatch(symmetry_element_target_i<N, T> &tgt) const
		throw(exception) = 0;
	//@}

};

/**	\brief Symmetry element dispatch target interface
	\tparam N Tensor order.
	\tparam T Tensor element type.

	The dispatch mechanism is used to determine the type of %symmetry
	element at runtime. See libtensor::symmetry_element_i<N, T> for
	more details about dispatching.

	This interface establishes the default target and is to be implemented
	by real targets. The default target is only called when a specific
	target is not implemented by a derived class.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class symmetry_element_target_i {
public:
	//!	\name Interface libtensor::symmetry_element_target_i<N, T>
	//@{

	/**	\brief Accepts the default dispatch
		\param elem Symmetry element.
	 **/
	virtual void accept_default(const symmetry_element_i<N, T> &elem)
		throw(exception) = 0;

	//@}
};


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_ELEMENT_IEX_H
