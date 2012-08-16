#ifndef LIBTENSOR_DIMENSIONS_H
#define LIBTENSOR_DIMENSIONS_H

#include "index_range.h"
#include "permutation.h"
#include "sequence.h"

#include "out_of_bounds.h"

namespace libtensor {


/** \brief Tensor %dimensions
    \tparam N Tensor order.

    Stores the number of %tensor elements along each dimension.

    \sa index_range

    \ingroup libtensor_core
 **/
template<size_t N>
class dimensions {
private:
    sequence<N, size_t> m_dims; //!< Tensor %dimensions
    sequence<N, size_t> m_incs; //!< Index increments
    size_t m_size; //!< Total size

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Converts a range of indexes to the dimensions object
        \param ir Index range.
     **/
    dimensions(const index_range<N> &ir);

    /** \brief Copy constructor
        \param d Another dimensions object.
     **/
    dimensions(const dimensions<N> &d);

    //@}


    //!    \name Dimensions manipulations, comparison, etc.
    //@{

    /** \brief Returns the total number of elements
     **/
    size_t get_size() const {
        return m_size;
    }

    /** \brief Returns the number of elements along a given dimension
     **/
    size_t get_dim(size_t i) const {
        return m_dims[i];
    }

    /** \brief Returns the linear increment along a given dimension
     **/
    size_t get_increment(size_t i) const {
        return m_incs[i];
    }

    /** \brief Returns true if an %index is within the %dimensions
     **/
    bool contains(const index<N> &idx) const;

    /** \brief Returns true if two %dimensions objects are equal
     **/
    bool equals(const dimensions<N> &other) const;

    /** \brief Permutes the %dimensions
        \return The reference to the current %dimensions object
     **/
    dimensions<N> &permute(const permutation<N> &p);

    //@}


    //!    \name Overloaded operators
    //@{

    /** \brief Returns the number of elements along a given dimension
     **/
    size_t operator[](size_t i) const {
        return get_dim(i);
    }

    /** \brief Returns true if two %dimensions objects are equal
     **/
    bool operator==(const dimensions<N> &other) const {
        return equals(other);
    }

    /** \brief Returns true if two %dimensions objects are different
     **/
    bool operator!=(const dimensions<N> &other) const {
        return !equals(other);
    }

    //@}


private:
    /** \brief Updates the linear increments for each dimension
     **/
    void update_increments();

};


/** \brief Tensor %dimensions (specialization for zero-order tensors)

    Stores the number of %tensor elements along each dimension.

    \ingroup libtensor_core
 **/
template<>
class dimensions<0> {
public:
    static const char *k_clazz; //!< Class name

public:
    //! \name Construction and destruction
    //@{

    /** \brief Converts a range of indexes to the dimensions object
        \param ir Index range.
     **/
    dimensions(const index_range<0> &ir) {

    }

    /** \brief Copy constructor
        \param d Another dimensions object.
     **/
    dimensions(const dimensions<0> &d) {

    }

    //@}


    //! \name Dimensions manipulations, comparison, etc.
    //@{

    /** \brief Returns the total number of elements
     **/
    size_t get_size() const {
        return 1;
    }

    /** \brief Returns the number of elements along a given dimension
     **/
    size_t get_dim(size_t i) const {

        static const char *method = "get_dim(size_t)";

        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "i");
    }

    /** \brief Returns the linear increment along a given dimension
     **/
    size_t get_increment(size_t i) const {

        static const char *method = "get_increment(size_t)";

        throw out_of_bounds(g_ns, k_clazz, method, __FILE__, __LINE__, "i");
    }

    /** \brief Returns true if two %dimensions objects are equal
     **/
    bool equals(const dimensions<0> &other) const {
        return true;
    }

    //@}


    //! \name Overloaded operators
    //@{

    /** \brief Returns the number of elements along a given dimension
     **/
    size_t operator[](size_t i) const {
        return get_dim(i);
    }

    /** \brief Returns true if two %dimensions objects are equal
     **/
    bool operator==(const dimensions<0> &other) const {
        return equals(other);
    }

    /** \brief Returns true if two %dimensions objects are different
     **/
    bool operator!=(const dimensions<0> &other) const {
        return !equals(other);
    }

    //@}


};


} // namespace libtensor

#endif // LIBTENSOR_DIMENSIONS_H
