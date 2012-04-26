#ifndef LIBTENSOR_BLOCK_TENSOR_CTRL_H
#define LIBTENSOR_BLOCK_TENSOR_CTRL_H

#include "../defs.h"
#include "../exception.h"
#include "block_tensor_i.h"
#include "block_tensor_ctrl.h"
#include "index.h"

namespace libtensor {

/** \brief Block %tensor control
    \tparam N Block %tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_core
**/
template<size_t N, typename T>
class block_tensor_ctrl {
private:
    block_tensor_i<N, T> &m_bt; //!< Controlled block %tensor

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates the control object
     **/
    block_tensor_ctrl(block_tensor_i<N, T> &bt);

    /** \brief Destroys the control object
     **/
    ~block_tensor_ctrl();

    //@}


    //!    \name Symmetry events
    //@{

    /** \brief Request to obtain the constant reference to the block
            %tensor's %symmetry
     **/
    const symmetry<N, T> &req_const_symmetry() throw(exception) {
        return m_bt.on_req_const_symmetry();
    }

    /** \brief Request to obtain the reference to the block %tensor's
            %symmetry
     **/
    symmetry<N, T> &req_symmetry() throw(exception) {
        return m_bt.on_req_symmetry();
    }

    //@}

    //!    \name Block events
    //@{
    dense_tensor_i<N, T> &req_block(const index<N> &idx) throw(exception);
    void ret_block(const index<N> &idx) throw(exception);
    dense_tensor_i<N, T> &req_aux_block(const index<N> &idx) throw(exception);
    void ret_aux_block(const index<N> &idx) throw(exception);
    bool req_is_zero_block(const index<N> &idx) throw(exception);
    void req_zero_block(const index<N> &idx) throw(exception);
    void req_zero_all_blocks() throw(exception);
    //@}

    //!    \name Synchronization events
    //@{
    void req_sync_on() throw(exception);
    void req_sync_off() throw(exception);
    //@}
};

template<size_t N, typename T>
inline block_tensor_ctrl<N, T>::block_tensor_ctrl(block_tensor_i<N, T> &bt) :
    m_bt(bt) {
}

template<size_t N, typename T>
block_tensor_ctrl<N, T>::~block_tensor_ctrl() {
}


template<size_t N, typename T>
inline dense_tensor_i<N, T> &block_tensor_ctrl<N, T>::req_block(const index<N> &idx)
    throw(exception) {

    return m_bt.on_req_block(idx);
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::ret_block(const index<N> &idx)
    throw(exception) {

    return m_bt.on_ret_block(idx);
}

template<size_t N, typename T>
inline dense_tensor_i<N, T> &block_tensor_ctrl<N, T>::req_aux_block(
    const index<N> &idx) throw(exception) {

    return m_bt.on_req_aux_block(idx);
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::ret_aux_block(const index<N> &idx)
    throw(exception) {

    return m_bt.on_ret_aux_block(idx);
}

template<size_t N, typename T>
inline bool block_tensor_ctrl<N, T>::req_is_zero_block(const index<N> &idx)
    throw(exception) {

    return m_bt.on_req_is_zero_block(idx);
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::req_zero_block(const index<N> &idx)
    throw(exception) {

    m_bt.on_req_zero_block(idx);
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::req_zero_all_blocks() throw(exception) {

    m_bt.on_req_zero_all_blocks();
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::req_sync_on() throw(exception) {

    m_bt.on_req_sync_on();
}

template<size_t N, typename T>
inline void block_tensor_ctrl<N, T>::req_sync_off() throw(exception) {

    m_bt.on_req_sync_off();
}

} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_CTRL_H

