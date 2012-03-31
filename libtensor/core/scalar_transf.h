#ifndef LIBTENSOR_SCALAR_TRANSF_H
#define LIBTENSOR_SCALAR_TRANSF_H

#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "index.h"

namespace libtensor {

/** \brief Transformation of a tensor element
    \tparam T Tensor element type.

    This template is a structure placeholder. It needs to be specialized for
    each %tensor element type.
    Any specialization of this class needs to provide:
    \code

    // Default constructor that creates the identity transformation
    scalar_transf();

    // Copy constructor
    scalar_transf(const scalar_transf<T> &tr);

    // Assignment operator
    scalar_transf<T> &operator=(const scalar_transf<T> &tr);

    // Reset the scalar transformation to identity
    void reset();

    // Apply scalar transformation st to this transformation
    scalar_transf<T> &transform(const scalar_transf<T> &st);

    // Invert this transformation
    scalar_transf<T> &invert();

    // Apply scalar transformation to tensor element x
    void apply(scalar_t &x) const;

    // Check if the transformation is the identity transformation
    bool is_identity() const;

    // Check if the transformation maps all elements to zero
    bool is_zero() const;

    bool operator==(const scalar_transf<T> &tr) const;

    bool operator!=(const scalar_transf<T> &tr) const;
    \endcode

    \ingroup libtensor_core
 **/
template<typename T>
class scalar_transf {
public:
    typedef T scalar_t;

};


} // namespace libtensor

#endif // LIBTENSOR_SCALAR_TRANSF_H
