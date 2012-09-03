#ifndef LIBTENSOR_GEN_BLOCK_TENSOR_CTRL_H
#define LIBTENSOR_GEN_BLOCK_TENSOR_CTRL_H

#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Generalized block tensor control (base)
    \tparam N Tensor order.
    \tparam Traits Block tensor traits.

    \sa gen_block_tensor_rd_ctrl, gen_block_tensor_wr_ctrl,
        gen_block_tensor_ctrl

    \ingroup libtensor_gen_block_tensor
**/
template<size_t N, typename Traits>
class gen_block_tensor_base_ctrl {
private:
    gen_block_tensor_base_i<N, Traits> &m_bt; //!< Controlled block tensor

public:
    /** \brief Initializes the control object
     **/
    gen_block_tensor_base_ctrl(gen_block_tensor_base_i<N, Traits> &bt) :
        m_bt(bt)
    { }

    /** \brief Turns on synchronization for thread safety
     **/
    void req_sync_on() {

        m_bt.on_req_sync_on();
    }

    /** \brief Turns off synchronization for thread safety
     **/
    void req_sync_off() {

        m_bt.on_req_sync_off();
    }

};


/** \brief Generalized read-only block tensor control
    \tparam N Tensor order.
    \tparam Traits Block tensor traits.

    \sa gen_block_tensor_base_ctrl, gen_block_tensor_wr_ctrl,
        gen_block_tensor_ctrl

    \ingroup libtensor_gen_block_tensor
**/
template<size_t N, typename Traits>
class gen_block_tensor_rd_ctrl :
    virtual public gen_block_tensor_base_ctrl<N, Traits> {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Type of read-only blocks
    typedef typename Traits::rd_block_type rd_block_type;

private:
    gen_block_tensor_rd_i<N, Traits> &m_bt; //!< Controlled block tensor

public:
    /** \brief Initializes the control object
     **/
    gen_block_tensor_rd_ctrl(gen_block_tensor_rd_i<N, Traits> &bt) :
        gen_block_tensor_base_ctrl<N, Traits>(bt),
        m_bt(bt)
    { }

    /** \brief Returns the constant reference to the block tensor's symmetry
            container
     **/
    const symmetry<N, element_type> &req_const_symmetry() {

        return m_bt.on_req_const_symmetry();
    }

    /** \brief Returns the read-only reference to a canonical block
        \param idx Index of the block.
        \return Reference to the requested block.
     **/
    rd_block_type &req_const_block(const index<N> &idx) {

        return m_bt.on_req_const_block(idx);
    }

    /** \brief Checks in a read-only canonical block
        \param idx Index of the block.
     **/
    void ret_const_block(const index<N> &idx) {

        m_bt.on_ret_const_block(idx);
    }

    /** \brief Returns true if a canonical block is zero
        \param idx Index of the block.
     **/
    bool req_is_zero_block(const index<N> &idx) {

        return m_bt.on_req_is_zero_block(idx);
    }

};


/** \brief Generalized read-write block tensor control
    \tparam N Tensor order.
    \tparam Traits Block tensor traits.

    \sa gen_block_tensor_base_ctrl, gen_block_tensor_rd_ctrl,
        gen_block_tensor_ctrl

    \ingroup libtensor_gen_block_tensor
**/
template<size_t N, typename Traits>
class gen_block_tensor_wr_ctrl :
    virtual public gen_block_tensor_base_ctrl<N, Traits> {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Type of read-write blocks
    typedef typename Traits::wr_block_type wr_block_type;

private:
    gen_block_tensor_wr_i<N, Traits> &m_bt; //!< Controlled block tensor

public:
    /** \brief Initializes the control object
     **/
    gen_block_tensor_wr_ctrl(gen_block_tensor_wr_i<N, Traits> &bt) :
        gen_block_tensor_base_ctrl<N, Traits>(bt),
        m_bt(bt)
    { }

    /** \brief Returns the reference to the block tensor's symmetry container
     **/
    symmetry<N, element_type> &req_symmetry() {

        return m_bt.on_req_symmetry();
    }

    /** \brief Returns the read-write reference to a canonical block
        \param idx Index of the block.
        \return Reference to the requested block.
     **/
    wr_block_type &req_block(const index<N> &idx) {

        return m_bt.on_req_block(idx);
    }

    /** \brief Checks in a canonical block
        \param idx Index of the block.
     **/
    void ret_block(const index<N> &idx) {

        return m_bt.on_ret_block(idx);
    }

    /** \brief Makes a canonical block zero
        \param idx Index of the block.
     **/
    void req_zero_block(const index<N> &idx) {

        m_bt.on_req_zero_block(idx);
    }

    /** \brief Make all blocks zero
     **/
    void req_zero_all_blocks() {

        m_bt.on_req_zero_all_blocks();
    }

};


/** \brief Generalized block tensor control
    \tparam N Tensor order.
    \tparam Traits Block tensor traits.

    \sa gen_block_tensor_base_ctrl, gen_block_tensor_rd_ctrl,
        gen_block_tensor_wr_ctrl

    \ingroup libtensor_gen_block_tensor
**/
template<size_t N, typename Traits>
class gen_block_tensor_ctrl :
    virtual public gen_block_tensor_rd_ctrl<N, Traits>,
    virtual public gen_block_tensor_wr_ctrl<N, Traits> {

private:
    gen_block_tensor_i<N, Traits> &m_bt; //!< Controlled block tensor

public:
    /** \brief Initializes the control object
     **/
    gen_block_tensor_ctrl(gen_block_tensor_i<N, Traits> &bt) :
        gen_block_tensor_base_ctrl<N, Traits>(bt),
        gen_block_tensor_rd_ctrl<N, Traits>(bt),
        gen_block_tensor_wr_ctrl<N, Traits>(bt),
        m_bt(bt)
    { }

    /** \brief Returns the read-write reference to an auxiliary (temporary)
            canonical block
        \param idx Index of the block.
        \return Reference to the requested block.
     **/
    wr_block_type &req_aux_block(const index<N> &idx) {

        return m_bt.on_req_aux_block(idx);
    }

    /** \brief Checks in an auxiliary (temporary) canonical block
        \param idx Index of the block.
     **/
    void ret_aux_block(const index<N> &idx) {

        return m_bt.on_ret_aux_block(idx);
    }

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BLOCK_TENSOR_CTRL_H
