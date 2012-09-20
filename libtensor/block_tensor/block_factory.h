#ifndef LIBTENSOR_BLOCK_FACTORY_H
#define LIBTENSOR_BLOCK_FACTORY_H

#include <libtensor/core/block_index_space.h>

namespace libtensor {


/** \brief Creates and destroys the blocks of a block tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Block Block type.

    \ingroup libtensor_gen_block_tensor
 **/
template<size_t N, typename T, typename Block>
class block_factory {
public:
    typedef Block block_type;

private:
    block_index_space<N> m_bis; //!< Block index space
    dimensions<N> m_bidims; //!< Block index dimensions

public:
    /** \brief Initializes the factory
        \param bis Block index space of the parent block tensor.
     **/
    block_factory(const block_index_space<N> &bis) :
        m_bis(bis), m_bidims(m_bis.get_block_index_dims())
    { }

    /** \brief Allocates a new block and returns its pointer
        \param idx Index of the block.
     **/
    block_type *create_block(const index<N> &idx) const {

        dimensions<N> dims = m_bis.get_block_dims(idx);
        return new block_type(dims);
    }

    /** \brief Destroys a block of a tensor
        \param blk Pointer to the block obtained from create_block().
     **/
    void destroy_block(block_type *blk) const {

        delete blk;
    }

};


} // namespace libtensor


#endif // LIBTENSOR_BLOCK_FACTORY_H
