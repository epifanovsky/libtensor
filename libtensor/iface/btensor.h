#ifndef LIBTENSOR_BTENSOR_H
#define LIBTENSOR_BTENSOR_H

#include "../defs.h"
#include "../exception.h"
#include "../core/block_index_space.h"
#include "../core/immutable.h"
#include <libtensor/block_tensor/block_tensor.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include "bispace.h"
#include "btensor_i.h"
#include "btensor_traits.h"
#include "labeled_btensor.h"

namespace libtensor {

/** \brief Base class for btensor
      \ingroup libtensor_iface
 **/
template<size_t N, typename T, typename Traits>
class btensor_base : public btensor_i<N, T>, public immutable {
private:
    typedef typename Traits::element_t element_t;
    typedef typename Traits::allocator_t allocator_t;

private:
    block_tensor<N, element_t, allocator_t> m_bt;
    block_tensor_ctrl<N, element_t> m_ctrl;

public:
    //!    \name Construction and destruction
    //@{
    /** \brief Constructs a block %tensor using provided information
            about blocks
        \param bi Information about blocks
     **/
    btensor_base(const bispace<N> &bis) :
         m_bt(bis.get_bis()), m_ctrl(m_bt) { }

    /** \brief Constructs a block %tensor using a block %index space
        \param bis Block %index space
     **/
    btensor_base(const block_index_space<N> &bis) :
        m_bt(bis), m_ctrl(m_bt) { }

    /** \brief Constructs a block %tensor using information about
            blocks from another block %tensor
        \param bt Another block %tensor
     **/
    btensor_base(const btensor_i<N, element_t> &bt) :
        m_bt(bt), m_ctrl(m_bt) { }

    /** \brief Virtual destructor
     **/
    virtual ~btensor_base() { }
    //@}

    //!    \name Implementation of block_tensor_i<N, T>
    //@{
    virtual const block_index_space<N> &get_bis() const;
    //@}

protected:
    //!    \name Implementation of libtensor::block_tensor_i<N,T>
    //@{
    virtual symmetry<N, T> &on_req_symmetry();
    virtual const symmetry<N, T> &on_req_const_symmetry();
    virtual dense_tensor_i<N, T> &on_req_const_block(const index<N> &idx);
    virtual void on_ret_const_block(const index<N> &idx);
    virtual dense_tensor_i<N, T> &on_req_block(const index<N> &idx);
    virtual void on_ret_block(const index<N> &idx);
    virtual dense_tensor_i<N, T> &on_req_aux_block(const index<N> &idx);
    virtual void on_ret_aux_block(const index<N> &idx);
    virtual bool on_req_is_zero_block(const index<N> &idx);
    virtual void on_req_zero_block(const index<N> &idx);
    virtual void on_req_zero_all_blocks();
    virtual void on_req_sync_on();
    virtual void on_req_sync_off();
    //@}

    //!    \name Implementation of libtensor::immutable
    //@{
    virtual void on_set_immutable();
    //@}
};

/** \brief User-friendly block %tensor

    \ingroup libtensor_iface
 **/
template<size_t N, typename T = double, typename Traits = btensor_traits<T> >
class btensor : public btensor_base<N, T, Traits> {
private:
    typedef typename Traits::element_t element_t;
    typedef typename Traits::allocator_t allocator_t;

public:
    btensor(const bispace<N> &bi) : btensor_base<N, T, Traits>(bi) { }
    btensor(const block_index_space<N> &bis) :
        btensor_base<N, T, Traits>(bis) { }
    btensor(const btensor_i<N, element_t> &bt) :
        btensor_base<N, T, Traits>(bt) { }
    virtual ~btensor() { }

    /** \brief Attaches a label to this %tensor and returns it as a
            labeled %tensor
     **/
    labeled_btensor<N, T, true> operator()(const letter_expr<N> &expr);

};


template<typename T, typename Traits>
class btensor<1, T, Traits> : public btensor_base<1, T, Traits> {
private:
    typedef typename Traits::element_t element_t;
    typedef typename Traits::allocator_t allocator_t;

public:
    btensor(const bispace<1> &bi) : btensor_base<1, T, Traits>(bi) { }
    btensor(const block_index_space<1> &bis) :
        btensor_base<1, T, Traits>(bis) { }
    btensor(const btensor_i<1, element_t> &bt) :
        btensor_base<1, T, Traits>(bt) { }
    virtual ~btensor() { }

    /** \brief Attaches a label to this %tensor and returns it as a
            labeled %tensor
     **/
    labeled_btensor<1, T, true> operator()(const letter &let);

    /** \brief Attaches a label to this %tensor and returns it as a
            labeled %tensor
     **/
    labeled_btensor<1, T, true> operator()(const letter_expr<1> &expr);

};


template<size_t N, typename T, typename Traits>
const block_index_space<N> &btensor_base<N, T, Traits>::get_bis() const {

    return m_bt.get_bis();
}


template<size_t N, typename T, typename Traits>
symmetry<N, T> &btensor_base<N, T, Traits>::on_req_symmetry() {

    return m_ctrl.req_symmetry();
}


template<size_t N, typename T, typename Traits>
const symmetry<N, T> &btensor_base<N, T, Traits>::on_req_const_symmetry() {

    return m_ctrl.req_const_symmetry();
}


template<size_t N, typename T, typename Traits>
dense_tensor_i<N, T> &btensor_base<N, T, Traits>::on_req_const_block(
    const index<N> &idx) {

    return m_ctrl.req_const_block(idx);
}


template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_ret_const_block(const index<N> &idx) {

    m_ctrl.ret_const_block(idx);
}


template<size_t N, typename T, typename Traits>
dense_tensor_i<N, T> &btensor_base<N, T, Traits>::on_req_block(
    const index<N> &idx) {

    return m_ctrl.req_block(idx);
}


template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_ret_block(const index<N> &idx) {

    m_ctrl.ret_block(idx);
}


template<size_t N, typename T, typename Traits>
dense_tensor_i<N, T> &btensor_base<N, T, Traits>::on_req_aux_block(
    const index<N> &idx) {

    return m_ctrl.req_aux_block(idx);
}


template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_ret_aux_block(const index<N> &idx) {

    m_ctrl.ret_aux_block(idx);
}


template<size_t N, typename T, typename Traits>
bool btensor_base<N, T, Traits>::on_req_is_zero_block(const index<N> &idx) {

    return m_ctrl.req_is_zero_block(idx);
}

template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_req_zero_block(const index<N> &idx) {

    m_ctrl.req_zero_block(idx);
}


template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_req_zero_all_blocks() {

    m_ctrl.req_zero_all_blocks();
}


template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_req_sync_on() {

    m_ctrl.req_sync_on();
}


template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_req_sync_off() {

    m_ctrl.req_sync_off();
}


template<size_t N, typename T, typename Traits>
void btensor_base<N, T, Traits>::on_set_immutable() {

    m_bt.set_immutable();
}


template<size_t N, typename T, typename Traits>
inline labeled_btensor<N, T, true> btensor<N, T, Traits>::operator()(
    const letter_expr<N> &expr) {

    return labeled_btensor<N, T, true>(*this, expr);
}


template<typename T, typename Traits>
inline labeled_btensor<1, T, true> btensor<1, T, Traits>::operator()(
    const letter &let) {

    return labeled_btensor<1, T, true>(*this, letter_expr<1>(let));
}


template<typename T, typename Traits>
inline labeled_btensor<1, T, true> btensor<1, T, Traits>::operator()(
    const letter_expr<1> &expr) {

    return labeled_btensor<1, T, true>(*this, expr);
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_H

