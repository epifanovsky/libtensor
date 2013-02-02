#ifndef LIBTENSOR_CUDA_BLOCK_TENSOR_CTRL_H
#define LIBTENSOR_CUDA_BLOCK_TENSOR_CTRL_H

#include "cuda_block_tensor_i.h"

namespace libtensor {

/** \brief Block %tensor control (base)
    \tparam N Block %tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_core
**/
template<size_t N, typename T>
class cuda_block_tensor_base_ctrl {
private:
    cuda_block_tensor_base_i<N, T> &m_bt; //!< Controlled block %tensor

public:
    /** \brief Creates the control object
     **/
    cuda_block_tensor_base_ctrl(cuda_block_tensor_base_i<N, T> &bt);

    /** \brief Destroys the control object
     **/
    virtual ~cuda_block_tensor_base_ctrl() { }

    /** \brief Request to obtain the constant reference to the block
            %tensor's %symmetry
     **/
    const symmetry<N, T> &req_const_symmetry();
};


/** \brief Block %tensor control (read-only)
    \tparam N Block %tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_core
**/
template<size_t N, typename T>
class cuda_block_tensor_rd_ctrl : virtual public cuda_block_tensor_base_ctrl<N, T> {
private:
    cuda_block_tensor_rd_i<N, T> &m_bt; //!< Controlled block %tensor

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates the control object
     **/
    cuda_block_tensor_rd_ctrl(cuda_block_tensor_rd_i<N, T> &bt);

    /** \brief Destroys the control object
     **/
    virtual ~cuda_block_tensor_rd_ctrl() { }

    //@}


    //!    \name Block events
    //@{
    dense_tensor_rd_i<N, T> &req_const_block(const index<N> &idx);
    void ret_const_block(const index<N> &idx);
    bool req_is_zero_block(const index<N> &idx);
    //@}

};


/** \brief Block %tensor control (read-write)
    \tparam N Block %tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_core
**/
template<size_t N, typename T>
class cuda_block_tensor_wr_ctrl : virtual public cuda_block_tensor_base_ctrl<N, T> {
private:
    cuda_block_tensor_wr_i<N, T> &m_bt; //!< Controlled block %tensor

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates the control object
     **/
    cuda_block_tensor_wr_ctrl(cuda_block_tensor_wr_i<N, T> &bt);

    /** \brief Destroys the control object
     **/
    virtual ~cuda_block_tensor_wr_ctrl() { }

    //@}


    //!    \name Symmetry events
    //@{

    /** \brief Request to obtain the reference to the block %tensor's
            %symmetry
     **/
    symmetry<N, T> &req_symmetry() {
        return m_bt.on_req_symmetry();
    }

    //@}

    //!    \name Block events
    //@{
    dense_tensor_wr_i<N, T> &req_block(const index<N> &idx);
    void ret_block(const index<N> &idx);
    void req_zero_block(const index<N> &idx);
    void req_zero_all_blocks();
    //@}

};


/** \brief Block %tensor control
    \tparam N Block %tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_core
**/
template<size_t N, typename T>
class cuda_block_tensor_ctrl :
    virtual public cuda_block_tensor_rd_ctrl<N, T>,
    virtual public cuda_block_tensor_wr_ctrl<N, T> {
private:
    cuda_block_tensor_i<N, T> &m_bt; //!< Controlled block %tensor

public:
    //!    \name Construction and destruction
    //@{

    /** \brief Creates the control object
     **/
    cuda_block_tensor_ctrl(cuda_block_tensor_i<N, T> &bt);

    /** \brief Destroys the control object
     **/
    virtual ~cuda_block_tensor_ctrl() { }

    //@}
};

template<size_t N, typename T>
inline cuda_block_tensor_base_ctrl<N, T>::cuda_block_tensor_base_ctrl(
        cuda_block_tensor_base_i<N, T> &bt) : m_bt(bt) {
}

template<size_t N, typename T>
inline cuda_block_tensor_rd_ctrl<N, T>::cuda_block_tensor_rd_ctrl(
        cuda_block_tensor_rd_i<N, T> &bt) :
        cuda_block_tensor_base_ctrl<N, T>(bt), m_bt(bt) {
}

template<size_t N, typename T>
inline cuda_block_tensor_wr_ctrl<N, T>::cuda_block_tensor_wr_ctrl(
        cuda_block_tensor_wr_i<N, T> &bt) :
        cuda_block_tensor_base_ctrl<N, T>(bt), m_bt(bt) {
}

template<size_t N, typename T>
inline cuda_block_tensor_ctrl<N, T>::cuda_block_tensor_ctrl(cuda_block_tensor_i<N, T> &bt) :
    cuda_block_tensor_base_ctrl<N, T>(bt),
    cuda_block_tensor_rd_ctrl<N, T>(bt),
    cuda_block_tensor_wr_ctrl<N, T>(bt),
    m_bt(bt) {
}


template<size_t N, typename T>
inline
const symmetry<N, T> &cuda_block_tensor_base_ctrl<N, T>::req_const_symmetry() {

    return m_bt.on_req_const_symmetry();
}

template<size_t N, typename T>
inline dense_tensor_rd_i<N, T> &cuda_block_tensor_rd_ctrl<N, T>::req_const_block(
    const index<N> &idx) {

    return m_bt.on_req_const_block(idx);
}

template<size_t N, typename T>
inline void cuda_block_tensor_rd_ctrl<N, T>::ret_const_block(const index<N> &idx) {

    return m_bt.on_ret_const_block(idx);
}

template<size_t N, typename T>
inline bool cuda_block_tensor_rd_ctrl<N, T>::req_is_zero_block(const index<N> &idx) {

    return m_bt.on_req_is_zero_block(idx);
}

template<size_t N, typename T>
inline dense_tensor_wr_i<N, T> &cuda_block_tensor_wr_ctrl<N, T>::req_block(
    const index<N> &idx) {

    return m_bt.on_req_block(idx);
}

template<size_t N, typename T>
inline void cuda_block_tensor_wr_ctrl<N, T>::ret_block(const index<N> &idx) {

    return m_bt.on_ret_block(idx);
}

template<size_t N, typename T>
inline void cuda_block_tensor_wr_ctrl<N, T>::req_zero_block(const index<N> &idx) {

    m_bt.on_req_zero_block(idx);
}

template<size_t N, typename T>
inline void cuda_block_tensor_wr_ctrl<N, T>::req_zero_all_blocks() {

    m_bt.on_req_zero_all_blocks();
}

} // namespace libtensor

#endif // LIBTENSOR_CUDA_BLOCK_TENSOR_CTRL_H

