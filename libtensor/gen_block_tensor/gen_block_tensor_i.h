#ifndef LIBTENSOR_GEN_BLOCK_TENSOR_I_H
#define LIBTENSOR_GEN_BLOCK_TENSOR_I_H

#include <libtensor/core/block_index_space.h>
#include <libtensor/core/symmetry.h>

namespace libtensor {


/** \brief Generalized block tensor base interface
    \tparam N Tensor order.
    \tparam Traits Block tensor traits.

    See gen_block_tensor_i for full documentation.

    \sa gen_block_tensor_rd_i, gen_block_tensor_wr_i, gen_block_tensor_i

    \ingroup libtensor_gen_block_tensor
 **/
template<size_t N, typename Traits>
class gen_block_tensor_base_i {
public:
    /** \brief Virtual destructor
     **/
    virtual ~gen_block_tensor_base_i() { }

    /** \brief Returns the block index space of the block tensor
     **/
    virtual const block_index_space<N> &get_bis() const = 0;

protected:
    /** \brief Turns on synchronization for thread safety
     **/
    virtual void on_req_sync_on() = 0;

    /** \brief Turns off synchronization for thread safety
     **/
    virtual void on_req_sync_off() = 0;

};


/** \brief Generalized block tensor read-only interface
    \tparam N Tensor order.
    \tparam Traits Block tensor traits.

    See gen_block_tensor_i for full documentation.

    \sa gen_block_tensor_base_i, gen_block_tensor_wr_i, gen_block_tensor_i

    \ingroup libtensor_gen_block_tensor
 **/
template<size_t N, typename Traits>
class gen_block_tensor_rd_i :
    virtual public gen_block_tensor_base_i<N, Traits> {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Type of read-only blocks
    typedef typename Traits::template rd_block_type<N>::type rd_block_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~gen_block_tensor_rd_i() { }

protected:
    /** \brief Request for the constant reference to the block tensor's symmetry
            container
     **/
    virtual const symmetry<N, element_type> &on_req_const_symmetry() = 0;

    /** \brief Request for the read-only reference to a canonical block
        \param idx Index of the block.
        \return Reference to the requested block.
     **/
    virtual rd_block_type &on_req_const_block(const index<N> &idx) = 0;

    /** \brief Invoked to return a read-only canonical block
        \param idx Index of the block.
     **/
    virtual void on_ret_const_block(const index<N> &idx) = 0;

    /** \brief Invoked to check whether a canonical block is zero
        \param idx Index of the block.
     **/
    virtual bool on_req_is_zero_block(const index<N> &idx) = 0;

};


/** \brief Generalized block tensor read-write interface
    \tparam N Tensor order.
    \tparam Traits Block tensor traits.

    See gen_block_tensor_i for full documentation.

    \sa gen_block_tensor_base_i, gen_block_tensor_rd_i, gen_block_tensor_i

    \ingroup libtensor_gen_block_tensor
 **/
template<size_t N, typename Traits>
class gen_block_tensor_wr_i :
    virtual public gen_block_tensor_base_i<N, Traits> {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Type of read-write blocks
    typedef typename Traits::template wr_block_type<N>::type wr_block_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~gen_block_tensor_wr_i() { }

protected:
    /** \brief Request for the reference to the block tensor's symmetry
            container
     **/
    virtual symmetry<N, element_type> &on_req_symmetry() = 0;

    /** \brief Request for the read-write reference to a canonical block
        \param idx Index of the block.
        \return Reference to the requested block.
     **/
    virtual wr_block_type &on_req_block(const index<N> &idx) = 0;

    /** \brief Invoked to return a canonical block
        \param idx Index of the block.
     **/
    virtual void on_ret_block(const index<N> &idx) = 0;

    /** \brief Invoked to make a canonical block zero
        \param idx Index of the block.
     **/
    virtual void on_req_zero_block(const index<N> &idx) = 0;

    /** \brief Invoked to make all blocks zero
     **/
    virtual void on_req_zero_all_blocks() = 0;


};


/** \brief Generalized block tensor interface
    \tparam N Tensor order.
    \tparam Traits Block tensor traits.

    The block tensor format assumes a divide-and-conquer approach to tensor
    storage and operations. Tensors are split along each mode into small blocks
    that are themselves tensors of the same order (number of dimensions).

    This generalized block tensor abstract class serves as a specification of
    the block tensor interface and protocol used by block tensor algorithms.

    \ingroup libtensor_gen_block_tensor
 **/
template<size_t N, typename Traits>
class gen_block_tensor_i :
    virtual public gen_block_tensor_rd_i<N, Traits>,
    virtual public gen_block_tensor_wr_i<N, Traits> {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Type of read-only blocks
    typedef typename Traits::template rd_block_type<N>::type rd_block_type;

    //! Type of read-write blocks
    typedef typename Traits::template wr_block_type<N>::type wr_block_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~gen_block_tensor_i() { }

protected:
    /** \brief Request for the read-write reference to an auxiliary (temporary)
            canonical block
        \param idx Index of the block.
        \return Reference to the requested block.
     **/
    virtual wr_block_type &on_req_aux_block(const index<N> &idx) = 0;

    /** \brief Invoked to return an auxiliary (temporary) canonical block
        \param idx Index of the block.
     **/
    virtual void on_ret_aux_block(const index<N> &idx) = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BLOCK_TENSOR_I_H
