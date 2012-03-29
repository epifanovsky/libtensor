#ifndef LIBTENSOR_SYMMETRY_ELEMENT_I_H
#define LIBTENSOR_SYMMETRY_ELEMENT_I_H

#include "../defs.h"
#include "../exception.h"
#include "block_index_space.h"
#include "index.h"
#include "mask.h"
#include "permutation.h"
#include "tensor_transf.h"

namespace libtensor {


/** \brief Symmetry element interface
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_core
 **/
template<size_t N, typename T>
class symmetry_element_i {
public:
    //!    \name Construction and destruction
    //@{

    /** \brief Virtual destructor
     **/
    virtual ~symmetry_element_i() { };

    //@}


    //!    \name Interface symmetry_element_i<N, T>
    //@{

    /** \brief Returns the type of symmetry
     **/
    virtual const char *get_type() const = 0;

    /** \brief Creates an identical copy of the %symmetry element
            using the new operator (the pointer must be deleted
            by the calling party)
     **/
    virtual symmetry_element_i<N, T> *clone() const = 0;

    /** \brief Checks whether the %symmetry element is applicable to
            the given block %index space
        \param bis Block %index space.
     **/
    virtual bool is_valid_bis(const block_index_space<N> &bis) const = 0;

    /** \brief Checks whether an %index is allowed by %symmetry
            (does not correspond to a zero block)
        \param idx Block %index.
     **/
    virtual bool is_allowed(const index<N> &idx) const = 0;

    /** \brief Applies the %symmetry element on an %index
        \param idx Block %index.
     **/
    virtual void apply(index<N> &idx) const = 0;

    /** \brief Applies the %symmetry element on an %index and
            transformation
        \param idx Block %index.
        \param tr Block transformation.
     **/
    virtual void apply(index<N> &idx, tensor_transf<N, T> &tr) const = 0;

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_ELEMENT_I_H
