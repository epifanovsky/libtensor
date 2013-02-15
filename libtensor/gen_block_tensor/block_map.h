#ifndef LIBTENSOR_BLOCK_MAP_H
#define LIBTENSOR_BLOCK_MAP_H

#include <utility> // for std::pair
#include <vector>
#include <libtensor/core/block_index_space.h>
#include <libtensor/core/immutable.h>

namespace libtensor {


/** \brief Stores pointers to blocks as an associative array
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam BtTraits Block tensor traits.

    The block map is an associative array of blocks with absolute indexes as
    keys. This class maintains such a map and provides facility to create and
    remove blocks. All the necessary memory management is done here as well.

    This implementation is not thread safe. Calls must be externally
        synchronized.

    \ingroup libtensor_gen_block_tensor
 **/
template<size_t N, typename BtTraits>
class block_map : public immutable {
public:
    typedef typename BtTraits::element_type element_type;
    typedef typename BtTraits::allocator_type allocator_type;
    typedef typename BtTraits::template block_type<N>::type block_type;
    typedef typename BtTraits::template block_factory_type<N>::type
        block_factory_type;
    typedef std::pair<size_t, block_type*> pair_type;

private:
    static const char k_clazz[]; //!< Class name

private:
    struct blkmap_cmp {
        bool operator()(
            const std::pair<size_t, block_type*> &p1,
            const std::pair<size_t, block_type*> &p2) {
            return p1.first < p2.first;
        }
    };

public:
    dimensions<N> m_bidims; //!< Block index dimensions
    block_factory_type m_bf; //!< Block factory
    mutable std::vector<pair_type> m_blocks; //!< Index to block mapping
    mutable bool m_sorted; //!< Whether the block map is sorted

public:
    /** \brief Constructs the map
        \param bis Block index space.
     **/
    block_map(const block_index_space<N> &bis) :
        m_bidims(bis.get_block_index_dims()), m_bf(bis), m_sorted(true)
    { }

    /** \brief Destroys the map and all the blocks
     **/
    virtual ~block_map();

    /** \brief Creates a block with the given index. If the block exists, it is
            removed and re-created
        \param idx Index of the block.
        \throw immut_violation If the object is immutable.
        \throw out_of_memory If there is not enough memory to create the block.
     **/
    void create(const index<N> &idx);

    /** \brief Removes a block
        \param idx Index of the block.
        \throw immut_violation If the object is immutable.
     **/
    void remove(const index<N> &idx);

    /** \brief Returns whether a block with a given index exists
        \param idx Index of the block.
     **/
    bool contains(const index<N> &idx) const;

    /** \brief Returns the absolute indexes of all contained blocks
        \param[out] blst List of indexes on output.
     **/
    void get_all(std::vector<size_t> &blst) const;

    /** \brief Returns the reference to a block identified by the index
        \param idx Index of the block.
        \throw block_not_found If the index supplied does not correspond
            to a block
     **/
    block_type &get(const index<N> &idx);

    /** \brief Removes all blocks
        \throw immut_violation If the object is immutable.
     **/
    void clear();

protected:
    virtual void on_set_immutable();

private:
    /** \brief Sorts the block mapping
     **/
    void sort() const;

    /** \brief Removes all blocks (without checking for immutability)
     **/
    void do_clear();

};


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_MAP_H
