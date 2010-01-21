#ifndef LIBTENSOR_SYMMETRY_ELEMENT_I_H
#define LIBTENSOR_SYMMETRY_ELEMENT_I_H

#include "../defs.h"
#include "../exception.h"
#include "block_index_space.h"
#include "index.h"
#include "mask.h"
#include "permutation.h"
#include "transf.h"

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

	/**	\brief Returns the type of symmetry
	 **/
	virtual const char *get_type() const = 0;

	/**	\brief Returns the mask of affected indexes
	 **/
	virtual const mask<N> &get_mask() const = 0;

	/**	\brief Adjusts the %symmetry element for a %permutation of
			%tensor indexes
	 **/
	virtual void permute(const permutation<N> &perm) = 0;

	/**	\brief Checks whether the %symmetry element is applicable to
			the given block %index space
		\param bis Block %index space.
	 **/
	virtual bool is_valid_bis(const block_index_space<N> &bis) const = 0;

	/**	\brief Checks whether an %index is allowed by %symmetry
			(does not correspond to a zero block)
		\param idx Block %index.
	 **/
	virtual bool is_allowed(const index<N> &idx) const = 0;

	/**	\brief Applies the %symmetry element on an %index
		\param idx Block %index.
	 **/
	virtual void apply(index<N> &idx) const = 0;

	/**	\brief Applies the %symmetry element on an %index and
			transformation
		\param idx Block %index.
		\param tr Block transformation.
	 **/
	virtual void apply(index<N> &idx, transf<N, T> &tr) const = 0;

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
