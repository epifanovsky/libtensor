#ifndef LIBTENSOR_SCALAR_TRANSF_H
#define LIBTENSOR_SCALAR_TRANSF_H

#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
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
    //! \name Constructors
    //@{

    scalar_transf();

    scalar_transf(const scalar_transf<T> &tr);

    //@}

    //! \name Manipulators
    //@{

    /** \brief Reset the scalar transformation to identity
     **/
	void reset();

	/** \brief Apply scalar transformation st to this transformation
	 **/
	scalar_transf<T> &transform(const scalar_transf<T> &st);

    /** \brief Invert this transformation
     **/
    scalar_transf<T> &invert();

    //@}

    /** \brief Apply scalar transformation to tensor element x
     **/
    void apply(scalar_t &x) const;

	/** \brief Check if the transformation is the identity transformation
	 **/
	bool is_identity() const;

	//! \name Comparison operators
	//@{
	bool operator==(const scalar_transf<T> &tr) const {
	    return true;
	}

	bool operator!=(const scalar_transf<T> &tr) const {
	    return (! operator==(tr));
	}
	//@}
};


template<typename T>
scalar_transf<T>::scalar_transf() {
    throw not_implemented(g_ns, "scalar_transf<T>",
            "scalar_transf()", __FILE__, __LINE__);
}


template<typename T>
scalar_transf<T>::scalar_transf(const scalar_transf<T> &tr) {
    throw not_implemented(g_ns, "scalar_transf<T>",
            "scalar_transf(const scalar_transf<T> &)", __FILE__, __LINE__);
}


} // namespace libtensor

#endif // LIBTENSOR_SCALAR_TRANSF_H
