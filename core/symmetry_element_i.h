#ifndef LIBTENSOR_SYMMETRY_ELEMENT_I_H
#define LIBTENSOR_SYMMETRY_ELEMENT_I_H

#include "defs.h"
#include "exception.h"
#include "index.h"
#include "mask.h"

namespace libtensor {


template<size_t N, typename T> class symmetry_element_target_i;


/**	\brief Symmetry element interface
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_core
 **/
template<size_t N, typename T>
class symmetry_element_i {
public:
	//!	\name Construction and destruction
	//@{

	/**	\brief Virtual destructor
	 **/
	virtual ~symmetry_element_i() { };

	//@}


	//!	\name Interface symmetry_element_i<N, T>
	//@{

	/**	\brief Returns the mask of affected indexes
	 **/
	virtual const mask<N> &get_mask() const = 0;

	/**	\brief Checks whether an %index is allowed by %symmetry
			(does not correspond to a zero block)
		\param idx Block %index.
	 **/
	virtual bool is_allowed(const index<N> &idx) const = 0;

	/**	\brief Applies the %symmetry element on an %index
		\param idx Block %index.
	 **/
	virtual void apply(index<N> &idx) const = 0;

	/**	\brief Checks whether two %symmetry elements are equal
			(have the same type and perform identically)
	 **/
	virtual bool equals(const symmetry_element_i<N, T> &se) const = 0;

	/**	\brief Creates an identical copy of the %symmetry element
			using the new operator (the pointer must be deleted
			by the calling party)
	 **/
	virtual symmetry_element_i<N, T> *clone() const = 0;

	virtual void dispatch(symmetry_element_target_i<N, T> &tgt) const
		throw(exception) = 0;

	/**	\brief Projects the %symmetry element to a higher-order
			space by creating a new %symmetry element object
			(the pointer must be deleted by the calling party)
		\param msk Mask that indicates which dimensions are to be
			projected.
		\param dims Dimensions in the new space.
		\throw exception If the projection cannot be performed.
	 **/
	//virtual symmetry_element_i<N + 1, T> *project_up(const mask<N + 1> &msk,
	//	const dimensions<N + 1> &dims) const throw(exception) = 0;

	/**	\brief Projects the %symmetry element to a lower-order space
			by creating a new %symmetry element object (the pointer
			must be deleted by the calling party)
		\param msk Mask that indicates which dimensions are to be
			preserved.
		\param dims Dimensions in the new space.
		\throw exception If the projection cannot be performed.
	 **/
	//virtual symmetry_element_i<N - 1, T> *project_down(
	//	const mask<N> &msk, const dimensions<N - 1> &dims) const
	//	throw(exception) = 0;
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

#endif // SYMMETRY_ELEMENT_I_H
