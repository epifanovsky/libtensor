#ifndef LIBTENSOR_SCALAR_TRANSF_H
#define LIBTENSOR_SCALAR_TRANSF_H

#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "index.h"

namespace libtensor {

/** \brief Transformation of a tensor element
    \tparam T Tensor element type.

    This template is a structure placeholder for element-wise transformation
    of tensor elements. It needs to be specialized for each %tensor element
    type. Any specialization of this class needs to provide:
    - the default constructor that creates the identity transformation
      \c scalar_transf();
    - a constructor that creates a transformation from a variable of the
      element type which will be a scalar multiplication with the respective
      value
      \c scalar_transf(const T &c);
      or
      \c scalar_transf(T c);
    - the copy constructor \c scalar_transf(const scalar_transf<T> &tr);
    - an assignment operator
      \c scalar_transf<T> &operator=(const scalar_transf<T> &tr);
    - the function
      \c void reset();
      to reset the scalar transformation to the identity transformation.
    - the function
      \c scalar_transf<T> &transform(const scalar_transf<T> &st);
      to transform the current transformation using another one.
    - the function
      \c scalar_transf<T> &invert();
      to invert the current transformation
    - the function
      \c void apply(scalar_t &x) const;
      to apply the scalar transformation to an element.
    - the function
      \c bool is_identity() const;
      to check if the current transformation maps any element onto itself
    - the function
      \c bool is_zero() const;
      to check if the current transformation maps all elements to zero
    - the functions
      \c bool operator==(const scalar_transf<T> &tr) const;
      and
      \c bool operator!=(const scalar_transf<T> &tr) const;
      to compare two transformations

    \ingroup libtensor_core
 **/
template<typename T>
class scalar_transf;


} // namespace libtensor

#endif // LIBTENSOR_SCALAR_TRANSF_H
