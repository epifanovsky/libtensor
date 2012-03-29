#ifndef LIBTENSOR_SCALAR_TRANSF_H
#define LIBTENSOR_SCALAR_TRANSF_H

#include "../defs.h"
#include "../exception.h"
#include "index.h"

namespace libtensor {

/**	\brief Transformation of a tensor element
	\tparam T Tensor element type.

	This template is a structure placeholder. It needs to be specialized for
	each %tensor element type.
	Any specialization of this class requires a default constructor that
	generates the identity transformation, as well as a copy constructor.

	\ingroup libtensor_core
 **/
template<typename T>
class scalar_transf {
public:
    typedef T scalar_t;

public:
    //! \name Manipulators
    //@{

    /** \brief Reset the scalar transformation to identity
     **/
	void reset();

	/** \brief Apply scalar transformation st to this transformation
	 **/
	void transform(const scalar_transf<T> &st);

    /** \brief Invert this transformation
     **/
    void invert();

    //@}

    /** \brief Apply scalar transformation to tensor element x
     **/
    void apply(scalar_t &x) const;

	/** \brief Check if the transformation is the identity transformation
	 **/
	bool is_identity() const;

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
