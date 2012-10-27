#ifndef LIBTENSOR_BLOCK_LIST_H
#define LIBTENSOR_BLOCK_LIST_H

#include <cstring> // for size_t
#include <set>
#include <vector>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/index.h>

namespace libtensor {


/** \brief Unique list of block in a block tensor

    XXX: In future: move to core part of library? base orbit_list on block_list?

    \ingroup libtensor_gen_bto
 **/
template<size_t N>
class block_list {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef typename std::set<size_t>::const_iterator iterator;

private:
    dimensions<N> m_dims;
    std::set<size_t> m_blks;

public:
    /** \brief Initializes an empty list
        \param bidims Block index dimensions.
     **/
    block_list(const dimensions<N> &bidims) :
        m_dims(bidims)
    { }

    /** \brief Initializes and fills the list
        \param bidims Block index dimensions.
        \param blst Vector of block indexes.
     **/
    block_list(const dimensions<N> &bidims, const std::vector<size_t> &blst) :
        m_dims(bidims), m_blks(blst.begin(), blst.end())
    { }

    /** \brief Add a block to the list
        \param absidx Absolute index of block
     **/
    void add(size_t absidx);

    /** \brief Add a block to the list
        \param idx Index of block
     **/
    void add(const index<N> &idx);

    /** \brief Erase a block from the list
        \param absidx Absolute index of block
     **/
    void erase(size_t absidx);

    /** \brief Erase a block from the list
        \param idx Index of block
     **/
    void erase(const index<N> &idx);

    /** \brief Erase a block from the list
        \param it Iterator pointing to element to be erased
     **/
    void erase(iterator it);

    /** \brief Empties the list
     **/
    void clear() {
        m_blks.clear();
    }

    /** \brief Return the current number of blocks in list
     **/
    size_t get_size() const { return m_blks.size(); }

    /** \brief Return the block index dimensions
     **/
    const dimensions<N> &get_dims() const {
        return m_dims;
    }

    /** \brief Does the list contain a given block
        \param absidx Absolute index of block to find in list
     **/
    bool contains(size_t absidx) const {
        return m_blks.find(absidx) != m_blks.end();
    }

    /** \brief Does the list contain a certain block
        \param idx Index of block to find in list
     **/
    bool contains(const index<N> &idx) const {
        return m_blks.find(abs_index<N>::get_abs_index(idx, m_dims))
                != m_blks.end();
    }

    /** \brief Iterator to first block in list
     **/
    iterator begin() const {
        return m_blks.begin();
    }

    /** \brief Iterator to end of list
     **/
    iterator end() const {
        return m_blks.end();
    }

    /** \brief Obtain absolute index of current block
     **/
    size_t get_abs_index(iterator &i) const {
        return *i;
    }

    /** \brief Obtain index of current block
        \param i Iterator
        \param[out] idx Index of block iterator points to
     **/
    void get_index(iterator &i, index<N> &idx) const {
        abs_index<N>::get_index(*i, m_dims, idx);
    }
};


template<size_t N>
const char *block_list<N>::k_clazz = "block_list<N>";


template<size_t N>
inline void block_list<N>::add(size_t absidx) {
#ifdef LIBTENSOR_DEBUG
    if (absidx >= m_dims.get_size()) {
        throw out_of_bounds(g_ns, k_clazz, "add(size_t)", __FILE__, __LINE__,
                "absidx");
    }
#endif

    m_blks.insert(absidx);
}


template<size_t N>
inline void block_list<N>::add(const index<N> &idx) {

    m_blks.insert(abs_index<N>::get_abs_index(idx, m_dims));
}


template<size_t N>
inline void block_list<N>::erase(size_t absidx) {
#ifdef LIBTENSOR_DEBUG
    if (absidx >= m_dims.get_size()) {
        throw out_of_bounds(g_ns, k_clazz, "erase(size_t)", __FILE__, __LINE__,
                "absidx");
    }
#endif

    m_blks.erase(absidx);
}


template<size_t N>
inline void block_list<N>::erase(const index<N> &idx) {

    m_blks.erase(abs_index<N>::get_abs_index(idx, m_dims));
}


template<size_t N>
inline void block_list<N>::erase(iterator it) {
    m_blks.erase(it);
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_LIST_H
