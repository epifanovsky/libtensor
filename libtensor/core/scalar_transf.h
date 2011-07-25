#ifndef LIBTENSOR_SCALAR_TRANSF_H
#define LIBTENSOR_SCALAR_TRANSF_H

#include "../defs.h"
#include "../exception.h"
#include "index.h"

namespace libtensor {

/**	\brief Transformation of a tensor element
	\tparam T Tensor element type.

	This template is a structure placeholder. It is to be specialized for
	each %tensor element type.
	Any specialization of this class requires a default constructor.

	\ingroup libtensor_core
 **/
template<typename T>
class scalar_transf {
public:
    //! \name Manipulators
    //@{
    /** \brief Reset the scalar transformation to identity
     **/
	void reset() { }

	/** \brief Apply scalar transformation st to this transformation
	 **/
	void transform(const scalar_transf<T> &st) { }

    /** \brief Invert this transformation
     **/
    void invert() { }

    //@}

	/** \brief Compute and return the inverse of this transformation
	 **/
	scalar_transf<T> inverse() const { }

    /** \brief Apply scalar transformation to tensor element x
     **/
    void apply(double &x) const { }

	/** \brief Check if the transformation is the identity
	 **/
	bool is_identity() const { return true; }

	//! \name Comparison operators
	//@{
	void operator==(const scalar_transf<T> &tr) const {
	    return true;
	}

	void operator!=(const scalar_transf<T> &tr) const {
	    return (! operator==(tr));
	}
	//@}
};


} // namespace libtensor

#endif // LIBTENSOR_SCALAR_TRANSF_H
