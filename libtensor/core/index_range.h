#ifndef LIBTENSOR_INDEX_RANGE_H
#define LIBTENSOR_INDEX_RANGE_H

#include "../defs.h"
#include "../exception.h"
#include "index.h"
#include "permutation.h"

namespace libtensor {

/** \brief Defines a range of %tensor elements

    Keeps the upper-left and the lower-right indexes that define a
    range.

    \ingroup libtensor_core
**/
template<size_t N>
class index_range {
private:
    index<N> m_begin; //!< Index of the first element of the range
    index<N> m_end; //!< Index of the last element of the range

public:
    /** \brief Creates a range using two indexes
        \param begin First %index of the range
        \param end Last %index of the range
        \throw exception if a range can't be defined by the given
            two indexes
    **/
    index_range(const index<N> &begin, const index<N> &end);

    /** \brief Copies a range from another index_range object
    **/
    index_range(const index_range<N> &r);

    /** \brief Returns the first %index of the range
    **/
    const index<N> &get_begin() const;

    /** \brief Returns the last %index of the range
    **/
    const index<N> &get_end() const;

    /** \brief Checks if two %index ranges are equal

        Two ranges are equal if they have the same beginning and the
        same end.
    **/
    bool equals(const index_range<N> &r) const;

    /** \brief Permutes both indices defining the range
    **/
    index_range<N> &permute(const permutation<N> &p);

};

template<size_t N>
inline index_range<N>::index_range(const index<N> &begin, const index<N> &end) :
    m_begin(begin), m_end(end) {

    for(size_t i=0; i != N; i++) if(m_begin[i] > m_end[i]) {
        size_t t = m_end[i];
        m_end[i] = m_begin[i]; m_begin[i] = t;
    }
}

template<size_t N>
inline index_range<N>::index_range(const index_range<N> &r) :
    m_begin(r.m_begin), m_end(r.m_end) {
}

template<size_t N>
inline const index<N> &index_range<N>::get_begin() const {
    return m_begin;
}

template<size_t N>
inline const index<N> &index_range<N>::get_end() const {
    return m_end;
}

template<size_t N>
inline bool index_range<N>::equals(const index_range<N> &r) const {
    return (m_begin.equals(r.m_begin) && m_end.equals(r.m_end));
}

template<size_t N>
inline index_range<N> &index_range<N>::permute(const permutation<N> &p) {
    m_begin.permute(p); m_end.permute(p);
    return *this;
}

} // namespace libtensor

#endif // LIBTENSOR_INDEX_RANGE_H

