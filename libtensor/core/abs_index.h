#ifndef LIBTENSOR_ABS_INDEX_H
#define LIBTENSOR_ABS_INDEX_H

#include "index.h"
#include "dimensions.h"

namespace libtensor {


/** \brief Absolute value of an %index within %dimensions
    \tparam N Tensor order.

    Tensor dimensions (number of elements along each side of the %tensor)
    define a space for indexes. Each index identifies a single %tensor element
    by providing its position along each dimension of the %tensor. The positions
    specified by the %index may not exceed %tensor dimensions.

    The absolute value of an index bound by dimensions is the position of
    the %tensor element referenced by the index in a linear array formed by
    vectorizing the %tensor. This vectorization is defined as unfolding the
    %tensor such that the last dimension of the %tensor is the fastest running
    and the first dimension is the slowest running.

    This class provides methods to work with the full index and its absolute
    value simultaneously, as well as static methods to perform conversion.

    \sa index, dimensions

    \ingroup libtensor_core
 **/
template<size_t N>
class abs_index {
public:
    static const char *k_clazz; //!< Class name

private:
    dimensions<N> m_dims; //!< Dimensions
    index<N> m_idx; //!< Index
    size_t m_aidx; //!< Absolute value of %index

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Initializes the first index within dimensions
        \param dims Dimensions.
     **/
    abs_index(const dimensions<N> &dims);

    /** \brief Initializes an index within dimensions
        \param idx Index.
        \param dims Dimensions.
     **/
    abs_index(const index<N> &idx, const dimensions<N> &dims);

    /** \brief Initializes an index by its absolute value within dimensions
        \param aidx Absolute value of the index.
        \param dims Dimensions.
     **/
    abs_index(size_t aidx, const dimensions<N> &dims);

    /** \brief Copy constructor
        \param other Another abs_index object.
     **/
    abs_index(const abs_index<N> &other);

    //@}


    //!    \name Manipulations
    //@{

    /** \brief Returns the %dimensions
     **/
    const dimensions<N> &get_dims() const {
        return m_dims;
    }

    /** \brief Returns the %index
     **/
    const index<N> &get_index() const {
        return m_idx;
    }

    /** \brief Returns the absolute %index
     **/
    size_t get_abs_index() const {
        return m_aidx;
    }

    /** \brief Increments the current %index, returns true if success
     **/
    bool inc();

    /** \brief Returns whether the current value is the last %index
            within the %dimensions
     **/
    bool is_last() const {
        return m_aidx + 1 >= m_dims.get_size();
    }

    /** \brief Increments the current %index, returns the reference to itself
     **/
    abs_index<N> &operator++() {
        inc();
        return *this;
    }

    //@}


    //! \name Absolute index evaluation (static methods)
    //@{

    /** \brief Computes the absolute value of the given %index within the given
            %dimensions
        \param idx Index (within %dimensions).
        \param dims Dimensions.
        \return Absolute value of the %index.
     **/
    static size_t get_abs_index(const index<N> &idx, const dimensions<N> &dims);

    /** \brief Computes the %index from its absolute value within the given
            %dimensions
        \param aidx Absolute value of the %index.
        \param dims Dimensions.
        \param[out] idx Index within the %dimensions.
     **/
    static void get_index(size_t aidx, const dimensions<N> &dims,
        index<N> &idx);

    //@}

};


} // namespace libtensor

#endif // LIBTENSOR_ABS_INDEX_H
